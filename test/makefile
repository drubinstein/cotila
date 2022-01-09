INCLUDES=../include
CXXFLAGS+=-I$(INCLUDES) -std=c++20 -Werror -Wall -Wextra -Wpedantic -Wno-missing-braces

test: test.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS)

.PHONY: clean

clean:
	rm -f test
