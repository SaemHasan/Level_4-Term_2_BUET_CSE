#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "lcgrand.h"

int main( void )
{
    srand(time(0));
    // lcgrandst(time(0), 100);
    float result = 0.0f;

    for( size_t i=0; i<10; i++ )
    {
        result = lcgrand(rand()%20);
        // result = logf( result ); // note change from `log()` to `logf()`
        printf( "%f\n", result );
    }
}