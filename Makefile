CXX = g++

UNAME = $(shell uname)

DEFINES =
INCLUDES = -Iexternal/densecrf/include
CXXFLAGS = -O2 -pedantic $(INCLUDES) $(DEFINES)
LDFLAGS = -Lexternal/densecrf/build/src
LDLIBS = -lz -ldensecrf

SOURCES := $(shell find src -name *.cpp)
OBJECTS := $(SOURCES:%=build/%.o)

all: mvcrf

mvcrf: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

build/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	$(RM) -rf $(OBJECTS) mvcrf

.PHONY: clean
