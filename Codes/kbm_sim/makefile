CC = gcc
CFLAGS = -g -Wall -Wextra -pedantic -Wno-vla-parameter

OBJ = main.o controller_kbm.o environment_kbm.o ctrl_pid_plus_stanley.o pid.o stanley.o utils.o
L_OBJ = controller_kbm.o environment_kbm.o

main.ex: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) -lm

controller_kbm.o: controller_kbm.c controller_kbm.h
	$(CC) $(CFLAGS) -c $<

environment_kbm.o: environment_kbm.c environment_kbm.h
	$(CC) $(CFLAGS) -c $<

ctrl_pid_plus_stanley.o: controllers_kbm/ctrl_pid_plus_stanley.c controllers_kbm/ctrl_pid_plus_stanley.h
	$(CC) $(CFLAGS) -c $<

stanley.o: controllers_kbm/controllers_basic/stanley.c controllers_kbm/controllers_basic/stanley.h
	$(CC) $(CFLAGS) -c $<

pid.o: ./controllers_kbm/controllers_basic/pid.c ./controllers_kbm/controllers_basic/pid.h
	$(CC) $(CFLAGS) -c $<

.PHONY: clean

clean:
	rm -f *.ex *.o

