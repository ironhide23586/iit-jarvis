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
    printf("Hello!\n");
    int N;
    scanf("%d", &N);
    int blocksReqd = ceil_h((float) (N * N) / (float) 192);

    printf("%d\n", blocksReqd);
}
