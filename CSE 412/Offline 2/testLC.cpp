#include <iostream>
#include <cmath>
#include "lcgrand.h"

int main( void )
{
    float result = 0.0f;

    for( size_t i=0; i<20; i++ )
    {
        result = lcgrand(1);
        // result = logf( result ); // note change from `log()` to `logf()`
        printf( "%f\n", result );
    }
}