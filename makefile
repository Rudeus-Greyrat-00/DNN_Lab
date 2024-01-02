CC = gcc
CFLAGS = -Wall
LIBS = -lm

PRGRNAME = main

all: main

main: main.c
	$(CC) $(CFLAGS) -o $(PRGRNAME) main.c $(LIBS)

clean:
	rm -f $(PRGRNAME)
