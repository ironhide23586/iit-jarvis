#include <stdio.h>
#include <math.h>

#define BLOCKS_PER_MP 8

int ceil_h(float f)
{
    int tmp = (int) f;
	if (f > tmp)
		tmp++;
	return tmp;
}

int main()
{
    int n = 11;
    int k = 11/2;
    k *= 2;

    for (k = 9; k <=2;k++)
    {
        printf("%d\n", k);
    }

    //printf("%d\n", k);
}
