// this is a classic problem solved by using threads
// there is a buffer of fixed size, which gets populated by a function - the producer
// then there is the consumer - which takes in data from the buffer

#include <iostream>
#include <thread>
#include <condition_variable>
#include <queue>

using namespace std;

queue<int> buffer; // the common buffer
const int BUF_SIZE = 50;
condition_variable cv;
mutex m;

void producer(int value)
{
    while(value)
    {
        unique_lock<mutex> l(m);
        cv.wait(l, [](){
            return (buffer.size() < BUF_SIZE);
        });

        cout << "producer : " << value << endl;
        buffer.push(value--);
        l.unlock();
        cv.notify_one();
    }
}

void consumer()
{
    while(1)
    {
        unique_lock<mutex> l(m);
        cv.wait(l, [](){
            return !buffer.empty();
        });

        int value = buffer.front();
        buffer.pop();
        std::cout << "consumer : " << value << endl;
        l.unlock();
        cv.notify_one();
        if (value == 1) break;
    }
}

int main()
{
    thread t1(producer, 100);
    thread t2(consumer);
    t1.join();
    t2.join();
    return 0;
}





