CC = gcc
CFLAGS = -g -Wall -Wextra -pedantic -Wno-vla-parameter

#OBJ = main.o environment_lin.o pid.o purepursuit.o utils.o
OBJ = main.o environment_lin.o pid.o stanley.o utils.o

main.ex: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $<

.PHONY: clean

clean:
	rm -f *.ex *.o

