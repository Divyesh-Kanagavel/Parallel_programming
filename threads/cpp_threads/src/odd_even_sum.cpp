#include <chrono>
#include <iostream>
#include <cstdint>
using namespace std;
using namespace chrono;

void findEvenSum(uint64_t start, uint64_t end, uint64_t& sum)
{
    sum = 0;
    for(uint64_t i=start; i < end;i++)
    {
        if ((i&1) == 0)
            sum += i;
    }
}

void findOddSum(uint64_t start, uint64_t end, uint64_t& sum)
{
    sum = 0;
    for(uint64_t i=start; i < end;i++)
    {
        if ((i&1) == 1)
            sum += i;
    }
}

int main()
{
    uint64_t start = 0;
    uint64_t end = 100000000;
    auto start_t = std::chrono::steady_clock::now();
    uint64_t sum_even;
    uint64_t sum_odd;
    findEvenSum(start, end, sum_even);
    findOddSum(start, end, sum_odd);
    auto end_t = std::chrono::steady_clock::now();

    std::cout << "even sum : " << sum_even << std::endl;
    std::cout << "odd sum : " << sum_odd << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t);
    std::cout << "Elapsed wall time: " << duration.count() << " milliseconds\n";
}