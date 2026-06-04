ZIG ?= zig
TARGET ?= x86_64-linux-musl
CPU ?= haswell
OPT ?= ReleaseFast

OUT := zig-out

.PHONY: all api lb build-index clean

all: api lb build-index

$(OUT):
	mkdir -p $@

api: $(OUT)
	$(ZIG) build-exe -target $(TARGET) -mcpu $(CPU) -O $(OPT) -lc -fsingle-threaded -fstrip \
		-femit-bin=$(OUT)/api src/main.zig
	rm -f api.o

lb: $(OUT)
	$(ZIG) build-exe -target $(TARGET) -mcpu $(CPU) -O $(OPT) -lc -fsingle-threaded -fstrip \
		-femit-bin=$(OUT)/lb src/lb.zig
	rm -f lb.o

build-index: $(OUT)
	$(ZIG) build-exe -target $(TARGET) -mcpu $(CPU) -O $(OPT) -lc -fstrip \
		-femit-bin=$(OUT)/build-index tools/build_index.zig
	rm -f build-index.o

clean:
	rm -rf $(OUT) .zig-cache *.o
