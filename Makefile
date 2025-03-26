all:
	gcc -Wall -Wextra -O2 -std=c11 -shared -fPIC -o libfusionflow_backend.so fusionflow_backend.c
