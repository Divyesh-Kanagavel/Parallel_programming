// thread pool : tasks distributed among threads with sync between threads and proper tasks completion

#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <functional>
#include <condition_variable>
#include <string>


using namespace std;
mutex cout_mutex;
class ThreadPool
{
    public:
        ThreadPool(size_t num_threads) : stop {false}
        {
            for(size_t i=0;i<num_threads;i++)
            {
                workers.emplace_back([this](){
                    for(;;)
                    {
                        unique_lock<mutex> lock(queue_mutex);
                        condition.wait(lock, [this](){
                            return (stop || !tasks.empty());
                        });
                        if (stop && tasks.empty()) return;
                        auto task = std::move(tasks.front());
                        tasks.pop();
                        lock.unlock();
                        //condition.notify_one();
                        task();
                    }
                });
            }
        }
        template <typename F>
        void enqueue(F&& task) //r-value reference
        {
            std::unique_lock<mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(task)); // forward function with same type of argument
            lock.unlock();
            condition.notify_one(); // notify any of the threads of this task
        }
        ~ThreadPool()
        {
            std::unique_lock<mutex> lock(queue_mutex);
            stop = true;
            lock.unlock();
            condition.notify_all();
            for (thread& worker : workers)
                worker.join();
        }

        private:
        std::vector<thread> workers;
        queue<function<void()>> tasks;
        condition_variable condition;
        bool stop;
        mutex queue_mutex;        
};

string get_thread_id()
{
    auto myid = this_thread::get_id();
    stringstream ss;
    ss << myid;
    string s = ss.str();
    return s;
}


int main()
{
    const int NUM_THREADS = 4; // usually it is better to set num_threads equal to number of cores to avoid context switching costs for large number of threads
    const int NUM_TASKS = 16;
    ThreadPool pl(NUM_THREADS);
    cout << "thread pool created ! \n";
    cout << "assign some tasks ! \n";
    for(int i=0;i<NUM_TASKS;i++)
    {
        pl.enqueue([i](){
            cout_mutex.lock();
            cout << "task " << i << " executed by thread " << get_thread_id() << endl;
            cout_mutex.unlock();
            this_thread::sleep_for(chrono::seconds(1)); // simulate work
        }
    ); // lambda function passed to pool
    }
    // main thread work
    this_thread::sleep_for(chrono::seconds(3));
    return 0;
}