#include <stdio.h>
#include <pthread.h>

pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
long balance = 0;

void* addMoney(void* arg) {
    long* money = (long *)arg;
    pthread_mutex_lock(&m);
    balance += *money;
    printf("balance after Money added: %ld\n", balance);
    pthread_cond_signal(&cond); // Wake up waiting threads
    pthread_mutex_unlock(&m);
    return NULL;
}

void* withdrawMoney(void* arg) {
    long* money = (long *)arg;
    pthread_mutex_lock(&m); // Lock before checking the condition
    while (balance <= 0) {  // Wait until enough balance
        printf("withdrawal thread waiting due to zero balance\n");
        pthread_cond_wait(&cond, &m);
    }
    if (balance >= *money)
    {
        balance -= *money;
        printf("after withdrawal, the balance is: %ld\n", balance);
    }
    else
    {
        printf("not sufficient balance \n");
    }
    pthread_mutex_unlock(&m); // Unlock after the operation
    return NULL;
}

int main() {
    pthread_t t1, t2;
    long with_amount = 300;
    long add_amount = 500;

    pthread_create(&t1, NULL, withdrawMoney, &with_amount);
    pthread_create(&t2, NULL, addMoney, &add_amount);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("balance after the transactions is: %ld\n", balance);
    pthread_mutex_destroy(&m);
    pthread_cond_destroy(&cond);

    return 0;
}
