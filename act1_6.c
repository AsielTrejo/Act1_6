#include <stdio.h>
#include <omp.h>
#include <math.h>


#ifndef M_PI
#    define M_PI 3.1415926535897932
#endif

#define NUM_THREADS 4

FILE *fptr2;
FILE *fptr4;
FILE *fptr6;
FILE *fptr8;

//int i;
double h,t,w,t2,k1,k2,k3,k4,ti1,x,y,func;
long N = 50000; //Numero de iteraciones del metodo numerico
double w0=M_PI/4,a=0,b=M_PI; //la funcion va a de "a" a "b".
                              //w0 es la condicion inicial


void func1RJK(int p){

   fptr2=fopen("RJ_n_1.txt","w");
   if(p){printf("Numero de pasos:%d Atendido por thread:%d\n", N,omp_get_thread_num());}
   fprintf(fptr2, "Datos del metodo de Runge-Kutta orden 4(variable ind.\t variable dep.\t numero de thread)\n");

      w = w0;
      fprintf(fptr2, "%f\t %f\n", a, w);

      for(int i=0;i<N;i++){

         h=(b-a)/N;
         t=a+(h*i);
         k1= h* ((t*exp(3.0*t))-(2.0*w));         
         x = t+(h/2.0);
         k2 = h * ((x*exp(3.0 * x ))-(2.0*(w+(0.5*k1))));
         k3 = h * ((x*exp(3.0 * x ))-(2.0*(w+(0.5*k2))));
         ti1 = a + (h*(i+1.0));
         k4 = h * ((ti1*exp(3.0*ti1))-(2.0*(w+k3)));
         w  = w + (1.0/6.0)*(k1+(2.0*k2)+(2.0*k3)+k4);
  
         fprintf(fptr2, "%f\t %f \tnumero de thread:%d \n", t+h, w,omp_get_thread_num());
      }
   fclose(fptr2);
}

void func2RJK(int p){

   fptr4=fopen("RJ_n_2.txt","w");
   if(p){printf("Numero de pasos:%d Atendido por thread:%d\n", N,omp_get_thread_num());}
   fprintf(fptr4, "Datos del metodo de Runge-Kutta orden 4(variable ind.\t variable dep.\t numero de thread)\n");
     
      w = w0;
      fprintf(fptr4, "%f\t %f\n", a, w);

      for(int i=0;i<N;i++){

         h=(b-a)/N;
         t=a+(h*i);
         func = 1.0 + pow(t-w,2.0);
         k1= h * func;
         x = t+(h/2.0);
         k2 = h * (1.0 + pow(x-(w+(0.5*k1)),2.0));
         k3 = h * (1.0 + pow(x-(w+(0.5*k2)),2.0));
         ti1 = a + (h * (i + 1.0));
         k4 = h * (1.0 + pow(ti1-(w+k3), 2.0));
         w  = w + (1.0/6.0)*(k1 + (2.0 * k2)+(2.0 * k3) + k4);
  
         fprintf(fptr4, "%f\t %f \tnumero de thread:%d  \n", t+h, w,omp_get_thread_num());
      }
   fclose(fptr4);
}

void func3RJK(int p){

   fptr6=fopen("RJ_n_3.txt","w");
   if(p){printf("Numero de pasos:%d Atendido por thread:%d\n", N,omp_get_thread_num());}
   fprintf(fptr6, "Datos del metodo de Runge-Kutta orden 4(variable ind.\t variable dep.\t numero de thread)\n");
     
      w = w0;
      fprintf(fptr6, "%f\t %f\n", a, w);

      for(int i=0;i<N;i++){

         h=(b-a)/N;
         t=a+(h*i);
         func = 1.0 + (w/t);
         k1= h * func;
         x = t+(h/2.0);
         k2 = h * (1+((w+(0.5*k1))/x));
         k3 = h * (1+((w+(0.5*k2))/x));
         ti1 = a + (h * (i + 1.0));
         k4 = h * (1+((w+k3)/ti1));
         w  = w + (1.0/6.0)*(k1 + (2.0 * k2)+(2.0 * k3) + k4);
  
         fprintf(fptr6, "%f\t %f \tnumero de thread:%d  \n", t+h, w,omp_get_thread_num());
      }
   fclose(fptr6);
}

void func4RJK(int p){

   fptr8=fopen("RJ_n_4.txt","w");
   if(p){printf("Numero de pasos:%d Atendido por thread:%d\n", N,omp_get_thread_num());}
   fprintf(fptr8, "Datos del metodo de Runge-Kutta orden 4(variable ind.\t variable dep.\t numero de thread)\n");
     
      w = w0;
      fprintf(fptr8, "%f\t %f\n", a, w);

      for(int i=0;i<N;i++){
         h=(b-a)/N;
         t=a+(h*i);
         func = cos(2.0*t*w)+sin(3.0*t*w);
         k1= h * func;
         x = t+(h/2.0);
         k2 = h * (cos(2.0*x*(w+(0.5*k1)))+sin(3.0*x*(w+(0.5*k1))));         
         k3 = h * (cos(2.0*x*(w+(0.5*k2)))+sin(3.0*x*(w+(0.5*k2))));        
         ti1 = a + (h * (i + 1.0));
         k4 = h * (cos(2.0*ti1*(w+k3))+sin(3.0*ti1*(w+k3)));
         w  = w + (1.0/6.0)*(k1 + (2.0 * k2)+(2.0 * k3) + k4);
  
         fprintf(fptr8, "%f\t %f \tnumero de thread:%d  \n", t+h, w,omp_get_thread_num());
      }
   fclose(fptr8);
}

int main()
{

double startTime = omp_get_wtime();
   func1RJK(0);
   func2RJK(0);
   func3RJK(0);
   func4RJK(0);
double endTime = omp_get_wtime();
printf("\nTiempo de ejecución serial %lf\n\n", (endTime - startTime));

omp_set_num_threads (NUM_THREADS);
startTime = omp_get_wtime();
#pragma omp parallel
{
   #pragma omp sections
   {
      #pragma omp section
         func1RJK(1);
      #pragma omp section
         func2RJK(1);
      #pragma omp section
         func3RJK(1);
      #pragma omp section
         func4RJK(1);
   }
}
   endTime = omp_get_wtime();
   printf("\nTiempo de ejecución en paralelo %lf\n\n", (endTime - startTime));
}
