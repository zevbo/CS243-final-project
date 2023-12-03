# the compiler: gcc for C program, define as g++ for C++
CC = clang++

# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings
CFLAGS  = -g -Wall --std=c++14 -fsanitize=undefined

# The build target 
TARGET = main
# FILES = layer.cpp main.cpp utils.cpp linear.cpp
FILES = model.cpp utils.cpp layer.cpp msl.cpp linear.cpp main.cpp relu.cpp

all: $(TARGET)

$(TARGET): $(TARGET).cpp
			$(CC) $(CFLAGS) -o $(TARGET) $(FILES)

clean:
			$(RM) $(TARGET)