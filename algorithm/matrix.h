#include <vector>
#include "bit_library.h"

#define vi std::vector<int>

using namespace std;

typedef struct matrix{
    int size, mod;
    vector<vi> element, origin;

    matrix(int _size, int _mod, int matrix_mod){
        size=_size;
        mod=_mod;
        element = vector<vi> (size, vi(size, 0));
        if(matrix_mod==1)
            for(int i=0;i<size;i++)
                element[i][i]=1;
        else if(matrix_mod==2)
            vector<vi> (size, vi(size, 1));
    }
    matrix(vector<vi> _element, int _mod){
        matrix(element.size(), _mod, -1);
        origin = _element;
    }

    void multiply(vector<vi> other){
        matrix ret = matrix(size, mod, 0);
        for(int i=0;i<size;i++){
            for(int j=0;j<size;j++){
                for(int k=0;k<size;k++){
                    ret.element[i][j] += element[i][k] * other[k][j];
                    ret.element[i][j] %= mod;
                }
            }
        }
        element = ret.element;
    }

    void multiply_K(int K){
        K = bit_reverse(K);
        while(K){
            multiply(element);
            if(K&1){
                multiply(origin);
            }
            K >>= 1;
        }
    }

    void print(){
        for(int i=0;i<size;i++){
            for(int j=0;j<size;j++)
                printf("%d ", element[i][j]);
            puts("");
        }
    }
} matrix;
