// producer consumer example with semaphores

#include <iostream>
#include <thread>
#include <semaphore>
#include <chrono>

using namespace std;

const int max_size = 5;
int arr[max_size];

binary_semaphore signal_producer {1}; // initially not locked
binary_semaphore signal_consumer {0}; // initally locked
// we want to start with producer always, hence the config

void producer()
{
    while(1)
    {
    signal_producer.acquire();
    cout << "producer : ";
    for(int i=0;i < max_size;i++)
    {
        arr[i] = i * i;
        cout << arr[i] << " " << std::flush;
        std::this_thread::sleep_for(chrono::milliseconds(200));
    }
    cout << endl;
    signal_consumer.release(); // release the semaphore for consumer use
}

}

void consumer()
{
    while(1){
    signal_consumer.acquire();
    cout << "consumer : " ;
    for(int i=max_size-1; i>=0; i--)
    {
        std::cout << arr[i] << " " << std::flush;
        arr[i] = 0;
        this_thread::sleep_for(chrono::milliseconds(200));
    }
    cout << endl;
    signal_producer.release();
}
}

int main()
{
    thread t1(producer);
    thread t2(consumer);
    t1.join();
    t2.join();
    return 0;
}