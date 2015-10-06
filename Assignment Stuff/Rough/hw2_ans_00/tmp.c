#include <stdio.h>
<<<<<<< HEAD
#include <unistd.h>

int main(int argc, char **argv)
{
    int numCPU = sysconf( _SC_NPROCESSORS_ONLN );
    int k = ceil((float)3/4);
    printf("%d, %d\n", numCPU, k);
=======

int main(int argc, char **argv)
{
    numCPU = sysconf( _SC_NPROCESSORS_ONLN );
>>>>>>> be208cb98833e9d10d0536d6e27269d7ea367232
}
