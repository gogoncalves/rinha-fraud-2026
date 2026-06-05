#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define MAX_BACKENDS 32
#define DEFAULT_ACCEPT_BATCH 128
#define DEFAULT_BACKLOG 4096
#define MAX_PREFIX 4096

/*
 * Layout of the SCM_RIGHTS message sent to each worker:
 *   iov[0] = uint16_t prefix_len  (little-endian on x86_64; native order, matching
 *            what the Zig worker uses)
 *   iov[1] = prefix_len bytes already drained from the client socket (or a single
 *            dummy byte when prefix_len == 0, so SEQPACKET keeps a non-zero
 *            payload — recvmsg returns r >= sizeof(u16)).
 *
 * This matches src/main.zig::onCtrlRecv which expects len_hdr in iov[0] and
 * the prefix buffer in iov[1].
 */
typedef struct {
    int fd;
    uint16_t len_hdr;
    uint8_t prefix[MAX_PREFIX];
    uint8_t dummy;
    struct iovec iov[2];
    union {
        struct cmsghdr cm;
        char buf[CMSG_SPACE(sizeof(int))];
    } control;
    struct msghdr msg;
    struct cmsghdr *cmsg;
} backend_t;

static int getenv_int(const char *name, int fallback) {
    const char *v = getenv(name);
    if (!v || !*v) return fallback;
    int parsed = atoi(v);
    return parsed > 0 ? parsed : fallback;
}

static int connect_backend(const char *path) {
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    int sndbuf = 256 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static void init_backend(backend_t *b, int fd) {
    memset(b, 0, sizeof(*b));
    b->fd = fd;
    b->dummy = 1;
    /* iov[0] always points at len_hdr; iov[1] base/len rewritten per send. */
    b->iov[0].iov_base = &b->len_hdr;
    b->iov[0].iov_len = sizeof(uint16_t);
    b->iov[1].iov_base = &b->dummy;
    b->iov[1].iov_len = 1;

    b->msg.msg_iov = b->iov;
    b->msg.msg_iovlen = 2;
    b->msg.msg_control = b->control.buf;
    b->msg.msg_controllen = sizeof(b->control.buf);
    b->cmsg = CMSG_FIRSTHDR(&b->msg);
    b->cmsg->cmsg_level = SOL_SOCKET;
    b->cmsg->cmsg_type = SCM_RIGHTS;
    b->cmsg->cmsg_len = CMSG_LEN(sizeof(int));
}

static int wait_for_socket(const char *path) {
    int tries = 0;
    while (tries++ < 600) {
        struct stat st;
        if (stat(path, &st) == 0) return 0;
        struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    return -1;
}

/*
 * Inline /ready short-circuit. Peeks the first bytes of the request without
 * consuming them and, if it looks like a GET /ready (no path suffix), replies
 * 200 OK directly from the LB and closes. Saves the SCM_RIGHTS cross + worker
 * wakeup for every Docker/contest health check. Reference: whereisanzi
 * crates/lb/src/main.rs::peek_is_health.
 */
static const char READY_RESP[] =
    "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";

static int try_serve_ready(int cfd) {
    char buf[64];
    ssize_t n = recv(cfd, buf, sizeof(buf), MSG_PEEK | MSG_DONTWAIT);
    if (n < (ssize_t)sizeof("GET /ready") - 1) return 0;
    if (memcmp(buf, "GET /ready", 10) != 0) return 0;
    /* Confirm boundary: next char must be ' ', '?', or end-of-buffer-not-yet-arrived
     * (we already peeked 10 bytes). The lb never serves other GET /ready* routes,
     * so accept space, '?', '\r' as the boundary. */
    char c = buf[10];
    if (c != ' ' && c != '?' && c != '\r' && c != '\n') return 0;
    const char *resp = READY_RESP;
    size_t left = sizeof(READY_RESP) - 1;
    while (left > 0) {
        ssize_t s = send(cfd, resp, left, MSG_NOSIGNAL);
        if (s > 0) { resp += s; left -= (size_t)s; continue; }
        if (s < 0 && errno == EINTR) continue;
        break;
    }
    return 1;
}

/*
 * Drain whatever bytes are already sitting in the TCP receive buffer for cfd.
 * Non-blocking; returns the number of bytes consumed (0..MAX_PREFIX). Anything
 * we read here MUST be forwarded — the worker will not see these bytes by
 * re-reading the fd, because we just consumed them.
 */
static size_t drain_ready_prefix(int cfd, uint8_t *buf) {
    /* Single recv: TCP_DEFER_ACCEPT guarantees data is present, and the worker's
     * onRecv loops until EAGAIN to pick up any tail. Saves a syscall per request,
     * which is hot on the LB (constrained to ~0.02 CPU under throttling). */
    ssize_t r = recv(cfd, buf, MAX_PREFIX, MSG_DONTWAIT);
    return r > 0 ? (size_t)r : 0;
}

static int send_fd_with_flags(backend_t *dst, int cfd, size_t prefix_len, int flags) {
    /* Rebuild controllen each send: MSG_CTRUNC etc. can stomp it. */
    dst->msg.msg_controllen = sizeof(dst->control.buf);
    dst->cmsg->cmsg_level = SOL_SOCKET;
    dst->cmsg->cmsg_type = SCM_RIGHTS;
    dst->cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(dst->cmsg), &cfd, sizeof(int));

    dst->len_hdr = (uint16_t)prefix_len;
    if (prefix_len > 0) {
        dst->iov[1].iov_base = dst->prefix;
        dst->iov[1].iov_len = prefix_len;
    } else {
        dst->iov[1].iov_base = &dst->dummy;
        dst->iov[1].iov_len = 1;
    }
    dst->msg.msg_iovlen = 2;

    for (;;) {
        ssize_t r = sendmsg(dst->fd, &dst->msg, MSG_NOSIGNAL | flags);
        if (r > 0) return 0;
        if (r < 0 && errno == EINTR) continue;
        return -1;
    }
}

static int send_fd(backend_t *dst, int cfd, size_t prefix_len) {
    return send_fd_with_flags(dst, cfd, prefix_len, MSG_DONTWAIT);
}

static int send_fd_blocking(backend_t *dst, int cfd, size_t prefix_len) {
    return send_fd_with_flags(dst, cfd, prefix_len, 0);
}

static int parse_backends(const char *env, char *paths[MAX_BACKENDS]) {
    int n = 0;
    char *tmp = strdup(env);
    char *save = NULL;
    char *tok = strtok_r(tmp, ",", &save);
    while (tok && n < MAX_BACKENDS) {
        /* trim surrounding whitespace */
        while (*tok == ' ' || *tok == '\t') tok++;
        size_t L = strlen(tok);
        while (L > 0 && (tok[L-1] == ' ' || tok[L-1] == '\t')) tok[--L] = 0;
        if (L > 0) paths[n++] = strdup(tok);
        tok = strtok_r(NULL, ",", &save);
    }
    free(tmp);
    return n;
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    signal(SIGPIPE, SIG_IGN);

    int port = 9999;
    if (getenv("LB_PORT")) port = atoi(getenv("LB_PORT"));
    int backlog = getenv_int("LB_BACKLOG", DEFAULT_BACKLOG);
    int accept_batch = getenv_int("LB_ACCEPT_BATCH", DEFAULT_ACCEPT_BATCH);
    const char *socks_env = getenv("API_SOCKETS");
    if (!socks_env || !*socks_env) socks_env = "/sock/api1.sock,/sock/api2.sock";

    char *paths[MAX_BACKENDS] = {0};
    int nb = parse_backends(socks_env, paths);
    if (nb <= 0) {
        fprintf(stderr, "[lb] no backends\n");
        return 2;
    }

    static backend_t backends[MAX_BACKENDS];
    for (int i = 0; i < nb; i++) {
        fprintf(stderr, "[lb] waiting %s\n", paths[i]);
        if (wait_for_socket(paths[i]) < 0) {
            fprintf(stderr, "[lb] timeout waiting %s\n", paths[i]);
            return 3;
        }
        int fd = -1;
        for (int t = 0; t < 100; t++) {
            fd = connect_backend(paths[i]);
            if (fd >= 0) break;
            struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 * 1000 * 1000 };
            nanosleep(&ts, NULL);
        }
        if (fd < 0) {
            fprintf(stderr, "[lb] connect failed %s\n", paths[i]);
            return 4;
        }
        init_backend(&backends[i], fd);
        fprintf(stderr, "[lb] connected %s (fd=%d)\n", paths[i], fd);
    }

    int lfd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (lfd < 0) { perror("socket"); return 5; }
    int on = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    setsockopt(lfd, SOL_SOCKET, SO_REUSEPORT, &on, sizeof(on));
    setsockopt(lfd, IPPROTO_TCP, TCP_DEFER_ACCEPT, &on, sizeof(on));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 6; }
    if (listen(lfd, backlog) < 0) { perror("listen"); return 7; }

    fprintf(stderr, "[lb] listening :%d backlog=%d batch=%d backends=%d\n",
            port, backlog, accept_batch, nb);

    int rr = 0;
    for (;;) {
        int accepted = 0;
        while (accepted < accept_batch) {
            int cfd = accept4(lfd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
            if (cfd < 0) {
                if (errno == EINTR) continue;
                break; /* EAGAIN/EWOULDBLOCK or other → exit batch */
            }
            accepted++;
            int one = 1;
            setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            setsockopt(cfd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));

            /* Short-circuit /ready at the LB so Docker/contest health checks
             * never wake an API worker. TCP_DEFER_ACCEPT guarantees data is
             * already in the receive buffer before accept returns. */
            if (try_serve_ready(cfd)) { close(cfd); continue; }

            int target = rr;
            rr = (rr + 1) % nb;

            /* Drain bytes already present (TCP_DEFER_ACCEPT means the request
             * is usually fully buffered) directly into the backend's prefix
             * staging area, then send fd + prefix in a single SCM_RIGHTS msg. */
            size_t prefix_len = drain_ready_prefix(cfd, backends[target].prefix);

            if (send_fd(&backends[target], cfd, prefix_len) != 0) {
                (void)send_fd_blocking(&backends[target], cfd, prefix_len);
            }
            close(cfd);
        }
        if (accepted == 0) {
            struct pollfd pfd = { .fd = lfd, .events = POLLIN, .revents = 0 };
            poll(&pfd, 1, -1);
        }
    }
}
