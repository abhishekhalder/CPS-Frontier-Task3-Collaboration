CC = gcc
CFLAGS = -g -Wall -Wextra -pedantic -Wno-vla-parameter
CLINKFLAGS_MPC =  -Wl,--rpath -Wl,/usr/local/lib
LIBS_MPC = `PKG_CONFIG_PATH=/usr/local/lib/pkgconfig: pkg-config --libs ipopt` -lstdc++ -lm
INCL_MPC = `PKG_CONFIG_PATH=/usr/local/lib/pkgconfig: pkg-config --cflags ipopt`

#OBJ = main.o controller_kbm.o environment_kbm.o ctrl_pid_plus_stanley.o pid.o stanley.o utils.o
OBJ = main.o controller_kbm.o environment_kbm.o ctrl_mpc.o ctrl_pid_plus_stanley.o pid.o stanley.o utils.o
#L_OBJ = controller_kbm.o environment_kbm.o

main.ex: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LIBS_MPC)
	#$(CC) $(CFLAGS) -o $@ $(OBJ) -lm

controller_kbm.o: controller_kbm.c controller_kbm.h
	$(CC) $(CFLAGS) -c $<

environment_kbm.o: environment_kbm.c environment_kbm.h
	$(CC) $(CFLAGS) -c $<

ctrl_mpc.o: controllers_kbm/ctrl_mpc.c controllers_kbm/ctrl_mpc.h
	$(CC) -O2 -DNDEBUG $(INCL_MPC) -c $<
	#$(CC) $(CLINKFLAGS_MPC) $(CFLAGS) -c $<
	#$(CC) $(CLINKFLAGS_MPC) $(CFLAGS) -c $< $(LIBS_MPC)
	#$(CC) $(CLINKFLAGS_MPC) $(CFLAGS) $(LIBS_MPC) -c $<

ctrl_pid_plus_stanley.o: controllers_kbm/ctrl_pid_plus_stanley.c controllers_kbm/ctrl_pid_plus_stanley.h
	$(CC) $(CFLAGS) -c $<

stanley.o: controllers_kbm/controllers_basic/stanley.c controllers_kbm/controllers_basic/stanley.h
	$(CC) $(CFLAGS) -c $<

pid.o: ./controllers_kbm/controllers_basic/pid.c ./controllers_kbm/controllers_basic/pid.h
	$(CC) $(CFLAGS) -c $<

.c.o:
	$(CC) -O2 -DNDEBUG $(INCL_MPC) -c -o $@ $<

.PHONY: clean

clean:
	rm -f *.ex *.o

