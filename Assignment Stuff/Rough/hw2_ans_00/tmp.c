#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    int numCPU = sysconf( _SC_NPROCESSORS_ONLN );
    int k = ceil((float)3/4);
    printf("%d, %d\n", numCPU, k);
}
