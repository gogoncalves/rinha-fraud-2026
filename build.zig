const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.resolveTargetQuery(.{
        .cpu_arch = .x86_64,
        .os_tag = .linux,
        .abi = .musl,
        .cpu_model = .{ .explicit = &std.Target.x86.cpu.haswell },
    });
    _ = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    const api = b.addExecutable(.{
        .name = "api",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .single_threaded = true,
            .strip = true,
            .link_libc = true,
        }),
    });
    b.installArtifact(api);

    const builder_tool = b.addExecutable(.{
        .name = "build-index",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/build_index.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    b.installArtifact(builder_tool);
}
