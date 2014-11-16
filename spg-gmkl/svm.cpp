#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <assert.h>
#include <limits>
#include "timer.hpp"
#include "svm.h"
//#include "sys/time.h"    // modified for windows

// modified for windows //
//////////////////////////
#ifndef NAN 					
#define NAN 0x7ff0000000000000 
#endif

#ifdef _WIN32
double cbrt(double x) {
	if (fabs(x) < DBL_EPSILON) return 0.0;
	if (x > 0.0) return pow(x, 1.0/3.0);
	return -pow(-x, 1.0/3.0);
}
#endif
//////////////////////////

int libsvm_version=LIBSVM_VERSION;
typedef float Qfloat;
typedef signed char schar;
#define LN2 log(2)
namespace LookUp{
	const int expmax = 30;
	const int precision = 6;
	const int len_lut = LookUp::expmax*(int)pow(10.0,LookUp::precision);
	const int mult = (int)pow(10.0,LookUp::precision);	
}
double *LUT = new double[LookUp::len_lut + 1];
void lookuptable(){
	double delta = 0.0;
	double add = powf(10,-LookUp::precision);
	for(int i=0;i<=LookUp::len_lut;i++){
		LUT[i] = exp(delta);
		delta+=add;
	}
}
void deleteLUT()
{
	delete [] LUT;
}
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst=new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}

int cubic(double A, double B, double C, double D,double *x);

static inline double powi(double base, int times)
{
	double tmp=base, ret=1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp=tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define DTAU 1e-5
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

namespace ProjGrad{
  const double alpha_min=0.01;
  const double alpha_max=10;
  const double gamma=1.0e-4;
  const double sigma1=0.1;
  const double sigma2=0.9;
  const double decay=0.5;
  const int M=10; 
  const int max_iter=2000;
  const double min_step=1.0e-64;

  // convergence tolerance 
  const double inf_abs_tol=1e-3;
  const double l2_abs_tol=1e-3;
  const double kkt_gap_tol=1e-3;

  const double loose_hz=1.0;
  const double tight_hz=0.1;
  const double good_ratio=0.9;
  const double bad_ratio=0.1;
}

namespace ReducedGradient{
  const double gold=(sqrt(5.0)+1.0)/2.0;
  const double gs_delta_init=0.1;
  const double step_clip=0.1;
}

namespace DaiFletcher{
  const double tol_r=1e-16;
  const double tol_lam=1e-15;
  const int max_iter=10000;
}

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
void (*svm_print_string) (const char *)=&print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

void copy_kernel(struct kernel *dest,const struct kernel *src)
{
	if(src==NULL || dest==NULL) return;
	dest->coef =src->coef;
	dest->scale_factor =src->scale_factor;	
	dest->kernel_type=src->kernel_type;
	dest->degree=src->degree;
	dest->gamma=src->gamma;
	dest->coef0=src->coef0;
	if(src->kernel_type==PRECOMPUTED)
	{
		dest->precomputed=new double*[src->precomputed_numrows];
		for(int i=0;i<src->precomputed_numcols;i++)
		{
			dest->precomputed[i]=new double[src->precomputed_numcols];
			memcpy(dest->precomputed[i],src->precomputed[i],sizeof(double)*src->precomputed_numcols);
		}
		dest->precomputed_numrows=src->precomputed_numrows;
		dest->precomputed_numcols=src->precomputed_numcols;
		dest->precomputed_filename=new char[strlen(src->precomputed_filename)+1];
		strcpy(dest->precomputed_filename,src->precomputed_filename);
	}
	else
	{
		dest->precomputed=NULL;
		dest->precomputed_numrows=-1;
		dest->precomputed_numcols=-1;
		dest->precomputed_filename=NULL;
	}
}

void copy_param(svm_parameter *new_p,const svm_parameter *param)
{
	if(param==NULL || new_p==NULL) return;

	new_p->svm_type=param->svm_type;
	new_p->d_regularizer=param->d_regularizer;
	new_p->d_proj=param->d_proj;
	new_p->solver_type=param->solver_type;
	new_p->num_kernels=param->num_kernels;
	new_p->L_p=param->L_p;
  new_p->l=param->l;
  
	new_p->kernels=new kernel[param->num_kernels];
	for(int i=0;i<param->num_kernels;i++)
		copy_kernel(&new_p->kernels[i],&param->kernels[i]);

	new_p->cache_size=param->cache_size;
	new_p->eps=param->eps;
	new_p->C=param->C;
	new_p->lambda=param->lambda;
	new_p->obj_threshold=param->obj_threshold;
	new_p->diff_threshold=param->diff_threshold;
	new_p->nr_weight=param->nr_weight;

	if(param->weight_label == NULL)
		new_p->weight_label=NULL;
	else
	{
		new_p->weight_label=Malloc(int,param->nr_weight);
		memcpy(new_p->weight_label,param->weight_label,sizeof(int)*param->nr_weight);
	}

	if(param->weight == NULL)
		new_p->weight=NULL;
	else
	{
		new_p->weight=Malloc(double,param->nr_weight);
		memcpy(new_p->weight,param->weight,sizeof(double)*param->nr_weight);
	}

	new_p->nu=param->nu;
	new_p->p=param->p;
	new_p->shrinking=param->shrinking;
	new_p->probability=param->probability;
}

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
	Cache(int l,long int size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int get_data(const int index, Qfloat **data, int len);
	void swap_index(int i, int j);	
private:
	int l;
	long int size;
	struct head_t
	{
		head_t *prev, *next;	// a circular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

Cache::Cache(int l_,long int size_):l(l_),size(size_)
{
	head=(head_t *)calloc(l,sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);
	size -= l * sizeof(head_t) / sizeof(Qfloat);
	size=max(size, 2 * (long int) l);	// cache must be large enough for two columns
	lru_head.next=lru_head.prev=&lru_head;
}

Cache::~Cache()
{
	for(head_t *h=lru_head.next; h != &lru_head; h=h->next)
		free(h->data);
	free(head);
}

void Cache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next=h->next;
	h->next->prev=h->prev;
}

void Cache::lru_insert(head_t *h)
{
	// insert to last position
	h->next=&lru_head;
	h->prev=lru_head.prev;
	h->prev->next=h;
	h->next->prev=h;
}

int Cache::get_data(const int index, Qfloat **data, int len)
{
	head_t *h=&head[index];
	if(h->len) lru_delete(h);
	int more=len - h->len;

	if(more > 0)
	{
		// free old space
		while(size < more)
		{
			head_t *old=lru_head.next;
			lru_delete(old);
			free(old->data);
			size+=old->len;
			old->data=0;
			old->len=0;
		}

		// allocate new space
		h->data=(Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
		size -= more;
		swap(h->len,len);
	}

	lru_insert(h);
	*data=h->data;
	return len;
}

void Cache::swap_index(int i, int j)
{
	if(i==j) return;

	if(head[i].len) lru_delete(&head[i]);
	if(head[j].len) lru_delete(&head[j]);
	swap(head[i].data,head[j].data);
	swap(head[i].len,head[j].len);
	if(head[i].len) lru_insert(&head[i]);
	if(head[j].len) lru_insert(&head[j]);

	if(i>j) swap(i,j);
	for(head_t *h=lru_head.next; h!=&lru_head; h=h->next)
	{
		if(h->len > i)
		{
			if(h->len > j)
				swap(h->data[i],h->data[j]);
			else
			{
				// give up
				lru_delete(h);
				free(h->data);
				size+=h->len;
				h->data=0;
				h->len=0;
			}
		}
	}
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const=0;
	virtual Qfloat *get_Qn(int n, int column, int len) const=0;
	virtual Qfloat *get_QD() const=0;
	virtual void swap_index(int i, int j) const=0;
	virtual ~QMatrix() {}
  virtual int get_num_kernels(void) const =0;
  virtual double *get_d(void) const=0;
};

class Kernel: public QMatrix {
public:
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();

	static double k_function(const svm_node *x, const svm_node *y,
				 const svm_parameter& param);
	virtual Qfloat *get_Q(int column, int len) const=0;
	virtual Qfloat *get_Qn(int n, int column, int len) const=0;
  
	virtual Qfloat *get_QD() const=0;

  int get_num_kernels(void) const { return num_kernels; }
  double *get_d(void) const { return d; } 
  double *get_scale_factor(void) const { return scale_factor; } 
	virtual void swap_index(int i, int j) const	// no so const...
	{
		swap(x[i],x[j]);
		if(x_square) swap(x_square[i],x_square[j]);
	}
protected:

  int l;
  Qfloat *QD;
  Qfloat **QD_all;
  int num_kernels;
	double *d;       // kernel coefficients
  
	Cache **cache;
	Cache *master_cache;
  
	double kernel_function(int i, int j) const
	{
		double sum=0.0;
		for(int n=0;n<num_kernels;n++)
    {
			if(d[n] > DTAU)
        sum+=kernel_all(n,i,j)*d[n];
    }
		return sum;
	}

	double kernel_all(int n, int i, int j) const
  {
    if(do_scale)
      return scale_factor[n]*kernel_all_unscaled(n, i, j);
    else
      return kernel_all_unscaled(n, i, j);
  }

private:
	const svm_node **x;
	double *x_square;

  bool do_scale;
  double* scale_factor;

	// svm_parameter
	struct kernel *kernels;
  static double dot(const svm_node *px, const svm_node *py);

	double kernel_all_unscaled(int n, int i, int j) const
	{
		switch (kernels[n].kernel_type)
		{
			case LINEAR:
				return dot(x[i],x[j]);
			case POLY:
				return powi(kernels[n].gamma*dot(x[i],x[j])+kernels[n].coef0,kernels[n].degree);
			case RBF:
				return exp(-kernels[n].gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
			case SIGMOID:
				return tanh(kernels[n].gamma*dot(x[i],x[j])+kernels[n].coef0);
			case PRECOMPUTED:
				return kernels[n].precomputed[(int)(x[i][0].value)][(int)(x[j][0].value)];
			default:
				return NAN;
		}
	}

};

Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
  :l(l), num_kernels(param.num_kernels)
{
	int do_rbf=0;
  kernels=new kernel[num_kernels];
	memcpy(kernels,param.kernels,sizeof(struct kernel)*num_kernels);
	d=new double[num_kernels];
  for(int n=0;n<num_kernels;n++)
	{
		d[n]=kernels[n].coef;
		if(kernels[n].kernel_type==RBF)
			do_rbf=1;
	}

	clone(x,x_,l);

	if(do_rbf)
	{
		x_square=new double[l];
		for(int i=0;i<l;i++)
			x_square[i]=dot(x[i],x[i]);
	}
	else
		x_square=0;

  if(num_kernels > 1)
  {
    do_scale=true;
    bool flag=false;
    scale_factor=new double[num_kernels];
    for(int n=0; n<num_kernels;n++)
    {
      scale_factor[n]=0;
      for(int i=0; i<l; i++)
        scale_factor[n]+=kernel_all_unscaled(n, i, i);
      assert(scale_factor[n] > 0);
      scale_factor[n]=1.0/scale_factor[n];
      if(fabs(scale_factor[n] - 1.0) > TAU)
        flag=true;
    }
    // Kernels are already unit trace normalized 
    if(!flag)
    {
      //delete [] scale_factor;
      do_scale=false;
    }
  }
  else
  {
    do_scale=false;
    scale_factor=new double[num_kernels];
    scale_factor[0]=1.0;
  }
}

Kernel::~Kernel()
{
	delete[] x;
	delete[] x_square;
  delete [] scale_factor; 
  delete[] QD;
  for(int i=0; i<l; i++)
    delete[] QD_all[i];
  delete[] QD_all;
  delete[] d;
  delete[] kernels;
  for(int n=0;n<num_kernels;n++)
    delete cache[n];    
  delete[] cache;
  delete master_cache;
}

double Kernel::dot(const svm_node *px, const svm_node *py)
{
	double sum=0;
	if(px->index == 0) ++px;
	if(py->index == 0) ++py;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum+=px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}

double Kernel::k_function(const svm_node *x, const svm_node *y,
			  const svm_parameter& param)
{
	double sum=0.0;
	for(int n=0;n<param.num_kernels;n++)
	{
		double wt=param.kernels[n].coef*param.kernels[n].scale_factor;
		switch(param.kernels[n].kernel_type)
		{
			case LINEAR:
				sum+=wt*dot(x,y);
				break;
			case POLY:
				sum+=wt*powi(param.kernels[n].gamma*dot(x,y)+param.kernels[n].coef0,param.kernels[n].degree);
				break;
			case RBF:
			{
				double tmp=0;
				const svm_node *x_tmp=x,*y_tmp=y;
				if(x_tmp->index == 0) ++x_tmp;
				if(y_tmp->index == 0) ++y_tmp;
				while(x_tmp->index != -1 && y_tmp->index !=-1)
				{
					if(x_tmp->index == y_tmp->index)
					{
						double diff=x_tmp->value - y_tmp->value;
						tmp+=diff*diff;
						++x_tmp;
						++y_tmp;
					}
					else
					{
						if(x_tmp->index > y_tmp->index)
						{	
							tmp+=y_tmp->value * y_tmp->value;
							++y_tmp;
						}
						else
						{
							tmp+=x_tmp->value * x_tmp->value;
							++x_tmp;
						}
					}
				}

				while(x_tmp->index != -1)
				{
					tmp+=x_tmp->value * x_tmp->value;
					++x_tmp;
				}

				while(y_tmp->index != -1)
				{
					tmp+=y_tmp->value * y_tmp->value;
					++y_tmp;
				}
				
				sum+=wt*exp(-param.kernels[n].gamma*tmp);
				break;
			}
			case SIGMOID:
				sum+=wt*tanh(param.kernels[n].gamma*dot(x,y)+param.kernels[n].coef0);
				break;
			case PRECOMPUTED:  //x: test (validation), y: SV
				sum+=wt*param.kernels[n].precomputed[(int)x->value][(int)y->value];
				break;
			default:
				return NAN;  // Unreachable 
		}
	}
	return sum;
}

class obj_grad_c {
public:
  obj_grad_c(double *d,double *grad_d,int num_kernels,double lambda):
    d(d), grad_d(grad_d), num_kernels(num_kernels),lambda(lambda){}
  
  virtual double primal()=0;
  virtual double dual()=0;
	virtual void grad(double* grad, double lambda)=0;
  virtual ~obj_grad_c() {}
protected:
  double *d;
  double *grad_d;
  int num_kernels;
  double lambda;
};

class obj_grad_ent: public obj_grad_c
{
public:
	obj_grad_ent(double *d,double *grad_d,int num_kernels,double lambda):
    obj_grad_c(d,grad_d,num_kernels, lambda) {}
  
	double primal()
	{
		double r=0.0;
		for(int n=0;n<num_kernels;n++)
		  if(d[n] > DTAU)
			  r+=d[n]*log(d[n]);
		return r;
	}

  double dual(){
    // Find max element 
    double exp_max=-INF; 
    double dsum=0.0;
    for(int n=0;n<num_kernels;n++)
      if(-grad_d[n] > exp_max) exp_max=-grad_d[n];
  
    // safe exponentiation 
    for(int n=0;n<num_kernels;n++)
      dsum+=exp((-grad_d[n] - exp_max)/lambda);
    
    return (exp_max/lambda) + log(dsum);
  }
  
	void grad(double* grad, double lambda)
	{
    for(int n=0;n<num_kernels;n++)
    {
      if(d[n] > DTAU)
        grad[n]=1.0 + log(d[n]);
      else
        // svnvish: BUGBUG
        // bandaid for now but seems to work reliably well
        grad[n]=1.0 + log(DTAU);
      grad[n] *= lambda;
    }
    return;
	}
};

class obj_grad_l1: public obj_grad_c
{
public:
	obj_grad_l1(double *d,double *grad_d,int num_kernels,double lambda):
    obj_grad_c(d,grad_d,num_kernels,lambda) {}
  
	double primal()
	{
		double r=0.0;
		for(int n=0;n<num_kernels;n++)
			if(d[n] > DTAU)
				r+=d[n];
		return 0.5*r*r;
	}
  
  double dual(){
    double r=-INF;
    for(int n=0;n<num_kernels;n++)
      if(r<grad_d[n]*grad_d[n])
        r=grad_d[n]*grad_d[n];
		return r/(2*lambda*lambda);
  }

	void grad(double* grad, double lambda)
	{
		double sum=0;
    for(int n=0;n<num_kernels;n++)
		sum+=d[n];
	for(int n=0;n<num_kernels;n++)
	      grad[n]=lambda*sum;
  
    return;
	}
};

class obj_grad_l2: public obj_grad_c
{
public:
  obj_grad_l2(double *d,double* grad_d,int num_kernels,double lambda):
    obj_grad_c(d,grad_d,num_kernels,lambda) {}

  double primal()
  {
    double r=0.0;
    for(int n=0;n<num_kernels;n++)
      if(d[n] > DTAU)
        r+=d[n]*d[n];
    return 0.5*r;
  }

  double dual(){
    double r=0.0;
    for(int n=0;n<num_kernels;n++)
      //if(grad_d[n] < -DTAU)
      r+=grad_d[n]*grad_d[n];
    return 0.5*r/(lambda*lambda);
  }

	void grad(double* grad, double lambda)
	{
    for(int n=0;n<num_kernels;n++)
      grad[n]=lambda*d[n];
    return;
	}
};

class obj_grad_lp: public obj_grad_c
{
public:
  //svnvish: BUGBUG
  //THRESH is somewhat arbitrary 
  obj_grad_lp(double *d,double* grad_d,int num_kernels,double lambda,float L_p,float L_q,double C):
    obj_grad_c(d,grad_d,num_kernels,lambda),L_p(L_p),L_q(L_q),
    THRESH(DTAU*10.0){
    info("THRESH=%.6f\n", THRESH);
  }
  
  double primal()
  {
    double r=0.0;
    for(int n=0;n<num_kernels;n++)
      if(d[n] > DTAU)
				r+=pow(d[n], (double) L_p);			
		return 0.5*pow(r, 2.0/L_p);
  }

  double dual(){
    double r=0.0;
    for(int n=0;n<num_kernels;n++)
      if(-grad_d[n]>DTAU)
				r+=pow(-grad_d[n], (double) L_q);	
		return 0.5*pow(r, 2.0/L_q)/(lambda*lambda);
  }

	void grad(double* grad, double lambda)
	{
		double r=0.0;
		for(int n=0;n<num_kernels;n++)
			if(d[n] > DTAU)
				r+=pow(d[n], (double) L_p);	
		r=pow(r, (L_p-2.0)/L_p);
    for(int n=0;n<num_kernels;n++)
      if(d[n]<THRESH)
        grad[n]=lambda*(d[n] < 0? -1:1)*pow(THRESH, L_p-1.0)/r;
      else
        grad[n]=lambda*(d[n] < 0? -1:1)*pow(d[n], L_p-1.0)/r;    
    return;
	}

private:
  float L_p;
  float L_q;
  double THRESH;
};

int project(double* x,const double* a,const double& b,const double* z,
            const double* l,const double* u,const int& max_iter,
            const int& n);

class proj_c {
public:
  proj_c(int num_kernels): num_kernels(num_kernels){}
  virtual void proj(double *d)=0;
  virtual void dd(double *d,double *g,double *dir,int max_idx)=0;
	virtual ~proj_c(void) {}
protected:
  int num_kernels;
};

// projection onto simplex
class proj_simplex: public proj_c
{
public:
	proj_simplex(int num_kernels):
    proj_c(num_kernels), a(0), l(0), u(0), z(0), b(1.0){
    a=new double[num_kernels];
    l=new double[num_kernels];
    u=new double[num_kernels];
    z=new double[num_kernels];
    for(int n=0;n<num_kernels;n++)
		{
			a[n]=1.0;
			l[n]=0.0;
			u[n]=1.0;
		}
  }
  
  ~proj_simplex(){
    delete [] a;
    delete [] l;
    delete [] u;
    delete [] z;
  }
  
	void proj(double *d)
	{
		for(int n=0;n<num_kernels;n++)
			z[n]=d[n];
    
		project(d, a, b, z, l, u, DaiFletcher::max_iter, num_kernels);
    return;
  }
  
	void dd(double *d,double *g,double *dir,int max_idx)
	{
	  // Compute direction of descent from reduced gradient
	  const double grad_max=g[max_idx];
  	double dirk_sum=0.0;
	  for(int n=0;n<num_kernels;n++)
  	{
	    dir[n]=0.0;
  	  if(d[n] >  DTAU || g[n]-grad_max < 0)
	    {
    	  dir[n]=grad_max-g[n];
  	    dirk_sum+=dir[n];
	    }
  	}
	  dir[max_idx]=-dirk_sum;
    return;
  }

private:
  double *a;
  double *l;
  double *u;
  double *z;
  double b;
  
};

// projection onto non-negative orthant
class proj_nn_orthant: public proj_c
{
public:
	proj_nn_orthant(int num_kernels):
    proj_c(num_kernels){}
  
	void proj(double *d)
	{
		for(int n=0;n<num_kernels;n++)
			if(d[n]<0) d[n]=0.0;
	}

	void dd(double *d,double *g,double *dir,int max_idx)
	{
		// Compute direction of descent from reduced gradient
		for(int n=0;n<num_kernels;n++)
		{
			dir[n]=0.0;
			if(d[n] >  DTAU || g[n] < 0)
				dir[n]=-g[n];
		}
    return;
	}
  
};
		

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha=\delta
//		y_i=+1 or -1
//		0 <= alpha_i <= Cp for y_i=1
//		0 <= alpha_i <= Cn for y_i=-1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
public:
	Solver() {};
	virtual ~Solver() { };

	struct SolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for Solver_NU
	};

  // svnvish: new functions introduced
  // begin
  
  // objective function calculations
	double primal_obj(bool reinit, obj_grad_c *obj_grad_fp);
	double dual_obj(obj_grad_c *obj_grad_fp);
  
  // gradient calculations
	double *grad(double *gradk, obj_grad_c *grad_fp);
	
  // end
  
	void Solve(int l, float L_p, const QMatrix& Q, const double *p_, const schar *y_,
             double *alpha_, double Cp, double Cn, double eps,
             SolutionInfo* si, int shrinking_, int solver_type, int d_regularizer, 
             int d_proj, double lambda_, double obj_threshold_, double diff_threshold_);

protected:
	int active_size;
	schar *y;
	double my_alpha;
	double obj_true;
  // svnvish: new variables introduced
  // begin
  int svm_calls;
  int svm_periter;
  int num_kernels;
  float L_q;
  float L_p;
  double *d;
  double *grad_d;
  double **Qalpha_all;
  double **Qalpha_bar_all;
  int shrinking;
  double lambda;         //tradeoff between regularizer and objective function
  double obj_threshold;  //threshold of the increase in objective function in line search
  double diff_threshold; //threshold that affects when line search terminates
  Timer fun_timer;       //store time spent in function evaluation 
  Timer solver_timer;
  // end

  // These quantities need to be updated after every update to d
  // begin
  double *G;		 // gradient of objective function
	double *G_bar; // gradient, if we treat free variables as 0
	Qfloat *QD;    // diagonal entries of the kernel matrix 
  // end

	enum { LOWER_BOUND, UPPER_BOUND, FREE };
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	double *alpha;
	const QMatrix *Q;
	double eps;
	int use_SPG;
	int monotone;
	int tune_precision;
	bool f1,f2;
	double Cp,Cn;
	double *p;
	int *active_set;
	int l;
	bool unshrink;	// XXX

	double get_C(int i)
	{
		return (y[i] > 0)? Cp : Cn;
	}
	void update_alpha_status(int i)
	{
		if(alpha[i] >= get_C(i))
			alpha_status[i]=UPPER_BOUND;
		else if(alpha[i] <= 0)
			alpha_status[i]=LOWER_BOUND;
		else alpha_status[i]=FREE;
	}
	bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
	bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
	bool is_free(int i) { return alpha_status[i] == FREE; }
	void swap_index(int i, int j);
	void reconstruct_gradient();
	virtual int select_working_set(int &i, int &j);
	virtual int qpqp_select_working_set(int &i, int &j);
	virtual int lpqp_select_working_set(int &i, int &j);
	virtual int ent_select_working_set(int &i, int &j);
	virtual double geteta(int i, int j, double &dir_grad);
	virtual double calculate_rho();
	virtual void do_shrinking();
  void init_Qalpha(void);
  void update_Qalpha(const int& i, 
                     const int& j, 
                     const double& delta_alpha_i,
                     const double& delta_alpha_j);
  
  // svnvish: begin
  // new functions introduced
  
  // compute gradients after d is updated
  void g_from_Qalpha(void);
  
  // compute d in case of entropic regularization 
  double d_ent(void);
  
  // compute d in case of L2 regularization 
  double d_l2(void);

  // compute d in case of Lp regularization 
  double d_lp(void);
  
  // end 

private:
	bool be_shrunk(int i, double Gmax1, double Gmax2);	

	void QPSolver(SolutionInfo* si);
  
  // svnvish: begin
  // new solvers added
  
  // SMO for entropy 
  void EntSolver(SolutionInfo* si);
  
  // SMO for L2 
  void QPQPSolver(SolutionInfo* si);

  // SMO for Lp
  void LPQPSolver(SolutionInfo* si);
  
  double compute_dir_hess(double* rr, int i, int j, int dir_i, int dir_j,double T2);

  // SPG-GMKL for Sum of kernels
  void SPGSolver(SolutionInfo* si, 
                  obj_grad_c *obj_grad_fp,
							proj_c *proj_fp);
  	
  // Mirror descent solver aka VSKL 
	void MirrorDescent(SolutionInfo* si, 
                     obj_grad_c *obj_grad_fp,
										 proj_c *proj_fp);
  
  // Reduced gradient solver aka SimpleMKL
  void ReducedGradient(SolutionInfo* si, 
                       obj_grad_c *obj_grad_fp,
											 proj_c *proj_fp);
  
  bool converged(const double* dk, 
                 double* gradk, 
                 const double primal,
                 obj_grad_c *obj_grad_fp,
                 proj_c *proj_fp);
  
  void normalize(double* gradk);
  
  void finalize(const double* dk,
                const double& obj, 
                SolutionInfo* si); 

  // end 

};


// Dispatch function
void Solver::Solve(int l_, float L_p_, const QMatrix& Q_, const double *p_, const schar *y_,
                   double *alpha_, double Cp_, double Cn_, double eps_,
                   SolutionInfo* si, int shrinking_, int solver_type, int d_regularizer, 
                   int d_proj, double lambda_, double obj_threshold_, double diff_threshold_) 
{
  
  // Keep track of time spent in the solver
  solver_timer.start();  
	tune_precision=1;use_SPG=1;monotone=0;
	l=l_;
  L_p=L_p_;
  // L_q is dual norm to L_p: 1/p + 1/q=1 => q=p/(p-1)
  L_q=(float)(L_p/(L_p - 1.0));
	Q=&Q_;
	QD=Q->get_QD();
  clone(p, p_,l);
	clone(y, y_,l);
	clone(alpha,alpha_,l);
	Cp=Cp_;
	Cn=Cn_;
	eps=eps_;
  shrinking=shrinking_;
  unshrink=false;
	lambda=lambda_;
	obj_threshold=obj_threshold_;
	diff_threshold=diff_threshold_;
  
  // initialize some information from the kernel class
  num_kernels=Q->get_num_kernels();
  d=Q->get_d();
  grad_d=new double[num_kernels];
  
	// initialize alpha_status
	{
		alpha_status=new char[l];
		for(int i=0;i<l;i++)
			update_alpha_status(i);
	}

	// initialize active set (for shrinking)
	{
		active_set=new int[l];
		for(int i=0;i<l;i++)
			active_set[i]=i;
		active_size=l;
	}

  // Allocate memory for Qalpha_all 
  Qalpha_all=new double*[l];
  for(int i=0;i<l;i++)
    Qalpha_all[i]=new double[num_kernels];

  // Allocate memory for Qalpha_bar_all
  Qalpha_bar_all=new double*[l];
  for(int i=0;i<l;i++)
    Qalpha_bar_all[i]=new double[num_kernels];
  
  init_Qalpha();

	// initialize gradient
	{
		G=new double[l];
		G_bar=new double[l];
    g_from_Qalpha();
  }

  si->obj=0;
  si->rho=0;
  
  if(num_kernels>1)
		if(solver_type == SPG || solver_type == SMKL || solver_type == MD || solver_type == SPGF || solver_type == PGD)
		{
			obj_grad_c *obj_grad_fp=0;
			proj_c *proj_fp=0;
      
			switch(d_regularizer)
			{
				case ENT:
					obj_grad_fp=new obj_grad_ent(d,grad_d,num_kernels,lambda);
					break;
				case L1:
					obj_grad_fp=new obj_grad_l1(d,grad_d,num_kernels,lambda);
					break;
				case L2:
					obj_grad_fp=new obj_grad_l2(d,grad_d,num_kernels,lambda);
					break;
				case LP:
					if(L_p != 2 && L_p != 1)
						obj_grad_fp=new obj_grad_lp(d,grad_d,num_kernels,lambda,L_p, L_q,max(fabs(Cp),fabs(Cn))/num_kernels);
					else if(L_p ==2)
						obj_grad_fp=new obj_grad_l2(d,grad_d,num_kernels,lambda);
					else
						obj_grad_fp=new obj_grad_l1(d,grad_d,num_kernels,lambda);						
					break;
			}

			switch(d_proj)
			{
				
        case SIMPLEX:
					proj_fp=new proj_simplex(num_kernels);
					break;
				case NN_ORTHANT:
					proj_fp=new proj_nn_orthant(num_kernels);
					break;
			}
	  if(solver_type == SPG){
		tune_precision=1;use_SPG=1;monotone=0;
		printf("SPG Solver Started\n");
        SPGSolver(si, obj_grad_fp, proj_fp);
		}
	else if(solver_type == SPGF){
		tune_precision=0;use_SPG=1;monotone=0;
        printf("SPGF Solver Started\n");
        SPGSolver(si, obj_grad_fp, proj_fp);
		}
	else if(solver_type == PGD){
		tune_precision=0;use_SPG=0;monotone=1;
        printf("PGD Solver Started\n");
        SPGSolver(si, obj_grad_fp, proj_fp);
		}	
      else if(solver_type == SMKL)
        ReducedGradient(si, obj_grad_fp, proj_fp);
      else if(solver_type == MD)
        MirrorDescent(si, obj_grad_fp, proj_fp);

			delete obj_grad_fp;
			delete proj_fp;
    }
		else
    {
      if(d_regularizer == L2 || (d_regularizer == LP && L_p == 2))
        QPQPSolver(si);
      else if(d_regularizer == LP)
        LPQPSolver(si);
      else
        EntSolver(si);
    }
  else
    QPSolver(si);
  
	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[active_set[i]]=alpha[i];
	}

  delete[] grad_d;
	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
  for(int i=0;i<l;i++)
    delete [] Qalpha_all[i];
  delete [] Qalpha_all;

	delete[] G_bar;
  for(int i=0;i<l;i++)
    delete [] Qalpha_bar_all[i];
  delete [] Qalpha_bar_all;
  solver_timer.stop();
  
  info("Time spent in solver: %f \n", solver_timer.total_cpu);
  
}

void Solver::QPSolver(SolutionInfo* si)
{
  
	// optimization step
	int iter=0;
	int counter=min(l,1000)+1;

	while(1)
	{    
		// show progress and do shrinking

		if(--counter == 0)
		{
			counter=min(l,1000);
			if(shrinking) do_shrinking();
			// info(".");
		}

		int i,j;
		if(select_working_set(i,j)!=0)
		{
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size=l;
			// info("*");
      if(select_working_set(i,j)!=0)
				break;
			else
				counter=1;	// do shrinking next iteration
		}
		
		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully		
		const Qfloat *Q_i=Q->get_Q(i,active_size);
		const Qfloat *Q_j=Q->get_Q(j,active_size);

		double C_i=get_C(i);
		double C_j=get_C(j);
    bool ui=is_upper_bound(i);
    bool uj=is_upper_bound(j);

		double old_alpha_i=alpha[i];
		double old_alpha_j=alpha[j];

		if(y[i]!=y[j])
		{
			double quad_coef=Q_i[i]+Q_j[j]+2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef=TAU;
			double delta=(-G[i]-G[j])/quad_coef;
			double diff=alpha[i] - alpha[j];
			alpha[i]+=delta;
			alpha[j]+=delta;
			
			if(diff > 0)
			{
				if(alpha[j] < 0)
				{
					alpha[j]=0;
					alpha[i]=diff;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i]=0;
					alpha[j]=-diff;
				}
			}
			if(diff > C_i - C_j)
			{
				if(alpha[i] > C_i)
				{
					alpha[i]=C_i;
					alpha[j]=C_i - diff;
				}
			}
			else
			{
				if(alpha[j] > C_j)
				{
					alpha[j]=C_j;
					alpha[i]=C_j + diff;
				}
			}
		}
		else
		{
			double quad_coef=Q_i[i]+Q_j[j]-2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef=TAU;
			double delta=(G[i]-G[j])/quad_coef;
			double sum=alpha[i] + alpha[j];
			alpha[i] -= delta;
			alpha[j]+=delta;

			if(sum > C_i)
			{
				if(alpha[i] > C_i)
				{
					alpha[i]=C_i;
					alpha[j]=sum - C_i;
				}
			}
			else
			{
				if(alpha[j] < 0)
				{
					alpha[j]=0;
					alpha[i]=sum;
				}
			}
			if(sum > C_j)
			{
				if(alpha[j] > C_j)
				{
					alpha[j]=C_j;
					alpha[i]=sum - C_j;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i]=0;
					alpha[j]=sum;
				}
			}
		}

		double delta_alpha_i=alpha[i] - old_alpha_i;
		double delta_alpha_j=alpha[j] - old_alpha_j;

    update_alpha_status(i);
    update_alpha_status(j);
    
    update_Qalpha(i, j, delta_alpha_i, delta_alpha_j);
        
		// update alpha_status and Qalpha_bar_all
		{
			update_alpha_status(i);
			update_alpha_status(j);
			int k, n;
			if(ui != is_upper_bound(i))
      {
        // svnvish: BUGBUG
        // bad memory access pattern
        for(n=0;n<num_kernels;n++)
        {
          const Qfloat *Q_i=Q->get_Qn(n,i,l);
          if(ui)
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n] -= C_i * Q_i[k];
          else
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n]+=C_i * Q_i[k];
        }
      }
      
			if(uj != is_upper_bound(j))
      {
        // svnvish: BUGBUG
        // bad memory access pattern
        for(n=0;n<num_kernels;n++)
        {
          const Qfloat *Q_j=Q->get_Qn(n,j,l);
          if(uj)
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n] -= C_j * Q_j[k];
          else
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n]+=C_j * Q_j[k];
        }
      }
		}
    
    // Update G and G_bar
    g_from_Qalpha();
    // also need to update QD
    QD=Q->get_QD();
    
  }
  
	// calculate rho

	si->rho=calculate_rho();

	// calculate objective value
	{
		double v=0;
		int i;
		for(i=0;i<l;i++)
			v+=alpha[i] * (G[i] + p[i]);

		si->obj=v/2;
	}

	// juggle everything back
	/*{
		for(int i=0;i<l;i++)
			while(active_set[i] != i)
				swap_index(i,active_set[i]);
				// or Q.swap_index(i,active_set[i]);
	}*/

	si->upper_bound_p=Cp;
	si->upper_bound_n=Cn;

  // info("\n QPSolver: optimization finished, #iter=%d\n",iter);
}

void Solver::QPQPSolver(SolutionInfo* si)
{
  // Find optimal d before getting into main loop 
  // Also compute initial objective function value 
  double alpha_sum=0.0;
  for(int i=0;i<l;i++)
    alpha_sum+=(alpha[i]*p[i]); 
  double obj=d_l2() + alpha_sum;
  info("initial obj=%f \n", -obj);

  // Update G and G_bar
  g_from_Qalpha();
  
	// optimization step
	int iter=0;
	int counter=min(l,1000)+1;

	while(1)
	{    
    // show progress and do shrinking
		if(--counter == 0)
		{
			counter=min(l,1000);
			if(shrinking) do_shrinking();
			info(".");
		}

		int i,j;
		if(qpqp_select_working_set(i,j)!=0)
		{
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size=l;
			info("*");
      if(qpqp_select_working_set(i,j)!=0)
				break;
			else
				counter=1;	// do shrinking next iteration
		}
    ++iter;

		// update alpha[i] and alpha[j], handle bounds carefully
		double C_i=get_C(i);
    double C_j=get_C(j);

    bool ui=is_upper_bound(i);
    bool uj=is_upper_bound(j);
    
		double old_alpha_i=alpha[i];
		double old_alpha_j=alpha[j];
    
    // Find a direction of descent 
		int dir_i=1;
		int dir_j=1;

    double dir_grad=0.0;
    double eta_max=0; // maximum step size 

    double *qq = new double[num_kernels];
    double *rr = new double[num_kernels];
    
    if(y[i] == y[j])
    {
      // dir_i and dir_j must have opposite signs
      // First try dir_i=+1 and dir_j=-1
      // else try dir_i=-1 and dir_j=+1
      if(G[i] - G[j] < G[j] - G[i])
      {
        dir_i=1;
        dir_j=-1;
        for(int n=0; n<num_kernels; n++)
          qq[n]=Qalpha_all[i][n] - Qalpha_all[j][n];
        dir_grad=G[i] - G[j];
        eta_max=min(C_i - alpha[i], alpha[j]);
      }
      else
      {
        dir_i=-1;
        dir_j=1;
        for(int n=0; n<num_kernels; n++)
          qq[n]=Qalpha_all[j][n] - Qalpha_all[i][n];
        dir_grad=G[j] - G[i];
        eta_max=min(alpha[i], C_j - alpha[j]);
      }
      for(int n=0;n<num_kernels;n++)
      {
        const Qfloat *Q_i=Q->get_Qn(n, i, active_size);
        const Qfloat *Q_j=Q->get_Qn(n, j, active_size);
        rr[n]=Q_i[i] + Q_j[j] - 2*Q_i[j];
      }
    }
    else
    {
      // dir_i and dir_j must have same sign
      // First try dir_i=+1 and dir_j=+1
      // else try dir_i=-1 and dir_j=-1
      if(G[i] + G[j] < -G[j] - G[i])
      {
        dir_i=1;
        dir_j=1;
        for(int n=0; n<num_kernels; n++)
          qq[n]=Qalpha_all[i][n] + Qalpha_all[j][n];
        dir_grad=G[i] + G[j];
        eta_max=min(C_i - alpha[i], C_j - alpha[j]);
      }
      else
      {
        dir_i=-1;
        dir_j=-1;
        for(int n=0; n<num_kernels; n++)
          qq[n]=-Qalpha_all[i][n] - Qalpha_all[j][n];
        dir_grad=-G[i] -G[j];
        eta_max=min(alpha[i], alpha[j]);
      }
      for(int n=0;n<num_kernels;n++)
      {
        const Qfloat *Q_i=Q->get_Qn(n, i, active_size);
        const Qfloat *Q_j=Q->get_Qn(n, j, active_size);
        rr[n]=Q_i[i] + Q_j[j] + 2*Q_i[j];
      }
    }
    assert(eta_max > 0);
        
    // Coefficient for \eta^4
    double aa=0.0;
    // Coefficient for \eta^3
    double bb=0.0;
    // Coefficient for \eta^2
    double cc=0.0;
    // Coefficient for \eta
    // theoretically dd=dir_grad 
    double dd=dir_grad; 
   
    for(int n=0; n<num_kernels; n++)
    {
      aa+=rr[n]*rr[n];
      bb+=rr[n]*qq[n];
      cc+=qq[n]*qq[n] - grad_d[n]*rr[n];
    }
    
    aa /= (8.0*lambda);
    bb /= (2.0*lambda);
    cc /= (2.0*lambda);

    double max_root=0;    
    if(aa == 0 && bb == 0 && cc == 0)
    {
      max_root=eta_max;
    }
    else
    {
      double roots[3];
      int num_roots=cubic(4*aa, 3*bb, 2*cc, dd, roots);
      assert(num_roots > 0);
      for(int k=0; k<num_roots;k++){
        if(roots[k] > eta_max)
          roots[k]=eta_max;
        if(roots[k] < 0)
          roots[k]=0;
        if(roots[k] > max_root)
          max_root=roots[k];
      }
      assert(max_root > 0);
    }
    double delta_alpha_i=dir_i*max_root;
    double delta_alpha_j=dir_j*max_root;
    
    alpha[i]=old_alpha_i + delta_alpha_i;
    alpha[j]=old_alpha_j + delta_alpha_j;
    
		// update alpha_status Qalpha and Qalpha_bar_all
		{
			update_alpha_status(i);
			update_alpha_status(j);
      update_Qalpha(i, j, delta_alpha_i, delta_alpha_j);
			int k, n;
			if(ui != is_upper_bound(i))
      {
        // svnvish: BUGBUG
        // bad memory access pattern
        for(n=0;n<num_kernels;n++)
        {
          const Qfloat *Q_i=Q->get_Qn(n,i,l);
          if(ui)
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n] -= C_i * Q_i[k];
          else
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n]+=C_i * Q_i[k];
        }
      }
      
			if(uj != is_upper_bound(j))
      {
        // svnvish: BUGBUG
        // bad memory access pattern
        for(n=0;n<num_kernels;n++)
        {
          const Qfloat *Q_j=Q->get_Qn(n,j,l);
          if(uj)
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n] -= C_j * Q_j[k];
          else
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n]+=C_j * Q_j[k];
        }
      }
		}
    
    // update alpha_sum, d, and obj 
    alpha_sum+=(p[i]*delta_alpha_i + p[j]*delta_alpha_j);
    obj=d_l2() + alpha_sum;
    
    // Update G, G_bar, and QD
    g_from_Qalpha();
    QD=Q->get_QD();
    
    info("iter=%d, obj=%f \n", iter, obj);
	delete [] qq;
	delete [] rr;
  }
  
	// calculate rho
	si->rho=calculate_rho();

	// calculate objective value
	{
    si->obj=d_l2();
    for(int i=0;i<l;i++)
      si->obj+=(alpha[i]*p[i]); 
    // Remember we minimized the -ve of the dual?
    si->obj *= -1;
  }
  
	si->upper_bound_p=Cp;
	si->upper_bound_n=Cn;

  // for(int i=0;i<l;i++)
  //   info("G[%d]=%f alpha=%f \n", i, G[i], alpha[i]); 
  
  info("\n QPQPSolver: optimization finished, #iter=%d\n",iter);
  return;
}

double Solver::compute_dir_hess(double* rr, int i, int j, int dir_i, int dir_j,double T2)
{
  // Compute directional Hessian
  double T0=0.0;
  for(int n=0;n<num_kernels;n++)
    if(d[n]>DTAU)
      T0+=(d[n]*rr[n]); 
  
  double T1=0.0;
  for(int n=0;n<num_kernels;n++)
    if(-grad_d[n]>DTAU)
      T1+=pow(-grad_d[n],(double) L_q);
  
  double T3=0.0;
  for(int n=0;n<num_kernels;n++)
  {
    double d_t_Qalpha=dir_i*Qalpha_all[i][n]+dir_j*Qalpha_all[j][n];
    if((d[n]>DTAU) && (-grad_d[n]>DTAU))
      T3-=d[n]*d_t_Qalpha*d_t_Qalpha/grad_d[n]; 
  }

  double dir_hess=0.0;
  if(T1>DTAU)
    dir_hess=T0+(2-L_q)*lambda*pow(T1,-2.0/L_q)*T2*T2+(L_q-1)*T3;
  else
    dir_hess=T0+(L_q-1)*T3;
  
  if(dir_hess < 0)
    dir_hess=DTAU;
  
  return dir_hess;
}

void Solver::LPQPSolver(SolutionInfo* si)
{
  // Find optimal d before getting into main loop 
  // Also compute initial objective function value 
  double alpha_sum=0.0;
  for(int i=0;i<l;i++)
    alpha_sum+=(alpha[i]*p[i]); 
  double obj=d_lp()+alpha_sum;
  info("initial obj=%f \n", -obj);

  // Update G and G_bar
  g_from_Qalpha();
  
  // Get diagonal kernel values 
  QD=Q->get_QD();
  
	// optimization step
	int iter=0;
	int counter=min(l,1000)+1;

  double *rr = new double[num_kernels];
  
	while(1)
	{    
    // show progress and do shrinking
		if(--counter == 0)
		{
			counter=min(l,1000);
			if(shrinking) do_shrinking();
			info(".");
		}

		int i,j;
		if(lpqp_select_working_set(i,j)!=0)
		{
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size=l;
			info("*");
      if(lpqp_select_working_set(i,j)!=0)
				break;
			else
				counter=1;	// do shrinking next iteration
		}
 
    ++iter;

    double C_i=get_C(i);
    double C_j=get_C(j);
    
    bool ui=is_upper_bound(i);
    bool uj=is_upper_bound(j);
    
		double old_alpha_i=alpha[i];
		double old_alpha_j=alpha[j];
    
    // Find a direction of descent 
		int dir_i=1;
		int dir_j=1;

    double dir_grad=0.0;
    double dir_grad_offset=0.0; 
    double eta_max=0; // maximum step size 

    
    if(y[i]==y[j])
    {
      // dir_i and dir_j must have opposite signs
      // First try dir_i=+1 and dir_j=-1
      // else try dir_i=-1 and dir_j=+1
      if(G[i]-G[j]<0)
      {
        dir_i=1;
        dir_j=-1;
        dir_grad=G[i]-G[j];
        dir_grad_offset=p[i]-p[j];
        eta_max=min(C_i-alpha[i], alpha[j]);
      }
      else
      {
        dir_i=-1;
        dir_j=1;
        dir_grad=G[j]-G[i];
        dir_grad_offset=p[j]-p[i];
        eta_max=min(alpha[i], C_j-alpha[j]);
      }
      for(int n=0;n<num_kernels;n++)
      {
        const Qfloat *Q_i=Q->get_Qn(n, i, active_size);
        const Qfloat *Q_j=Q->get_Qn(n, j, active_size);
        rr[n]=Q_i[i]+Q_j[j]-2*Q_i[j];
      }
    }
    else
    {
      // dir_i and dir_j must have same sign
      // First try dir_i=+1 and dir_j=+1
      // else try dir_i=-1 and dir_j=-1
      // if(G[i]+G[j]<-(G[i]+G[j]))
      if(G[i]+G[j]<0)
      {
        dir_i=1;
        dir_j=1;
        dir_grad=G[i]+G[j];
        dir_grad_offset=p[i]+p[j];
        eta_max=min(C_i-alpha[i], C_j-alpha[j]);
      }
      else
      {
        dir_i=-1;
        dir_j=-1;
        dir_grad=-G[i]-G[j];
        dir_grad_offset=-p[i]-p[j];
        eta_max=min(alpha[i], alpha[j]);
      }
      for(int n=0;n<num_kernels;n++)
      {
        const Qfloat *Q_i=Q->get_Qn(n, i, active_size);
        const Qfloat *Q_j=Q->get_Qn(n, j, active_size);
        rr[n]=Q_i[i]+Q_j[j]+2*Q_i[j];
      }
    }
    assert(dir_grad<0);
    assert(eta_max>0);
    
    double dir_hess=compute_dir_hess(rr,i,j,dir_i,dir_j,dir_grad-dir_grad_offset);

    double old_obj=obj;    
    double old_dir_grad=dir_grad;
    double eta_min=0.0;
    double eta=-dir_grad/dir_hess;

    if(eta > eta_max)
      eta=eta_max;
    if(eta < eta_min)
      eta=eta_min+0.5*(eta_max-eta_min);
    int ls_iter=0;

    while(1)
    {
      double new_alpha_i=old_alpha_i+eta*dir_i;
      double new_alpha_j=old_alpha_j+eta*dir_j;
      double change_i=new_alpha_i-alpha[i];
      double change_j=new_alpha_j-alpha[j];      
      alpha[i]=new_alpha_i;
      alpha[j]=new_alpha_j;
      update_alpha_status(i);
      update_alpha_status(j);
      update_Qalpha(i, j, change_i, change_j);
      alpha_sum+=(p[i]*change_i + p[j]*change_j); 
      obj=d_lp()+alpha_sum;      

      // Compute new directional gradient
      dir_grad=dir_grad_offset;
      for(int n=0;n<num_kernels;n++)
        if(d[n] > DTAU)
          dir_grad+=d[n]*(dir_i*Qalpha_all[i][n]+dir_j*Qalpha_all[j][n]);
      
      if((obj < old_obj + obj_threshold*eta*old_dir_grad) ||
         (fabs(eta_max - eta_min) < diff_threshold) ||
         (fabs(dir_grad) < diff_threshold) ||
         (fabs(eta - eta_max) < diff_threshold && dir_grad < 0))
      {
        if((obj > old_obj) || (eta <= 0))
        {
          info("\n iter=%d ls_iter=%d \n", iter, ls_iter);
          info("eta=%f eta_max=%f eta_min=%f \n", eta, eta_max, eta_min);
          info("old_obj=%f obj=%f old_dir_grad=%f dir_grad=%f \n", old_obj, obj, old_dir_grad, dir_grad);
        }
        break;
      }
			
      if(dir_grad > 0)
        eta_max=eta;
      else
      {
        if(obj > old_obj)
        {
          info("iter=%d ls_iter=%d \n", iter, ls_iter);
          info("eta=%f eta_max=%f eta_min=%f \n", eta, eta_max, eta_min);
          info("old_obj=%f obj=%f old_dir_grad=%f dir_grad=%f \n", old_obj, obj, old_dir_grad, dir_grad);
        }
        
        if(eta >= eta_min+TAU)
          assert(obj <= old_obj);
        
        eta_min=eta;
      }
      
      // Try a Newton-Raphson step
      dir_hess=compute_dir_hess(rr,i,j,dir_i,dir_j,dir_grad-dir_grad_offset); 
      eta=eta-dir_grad/dir_hess;
      // If the result is outside the brackets, use bisection
      if(eta > eta_max || eta <= eta_min)
        eta=eta_min + 0.5*(eta_max - eta_min);
      
      ls_iter++;
			
      if(ls_iter > 20)
        info("iter=%d ls_iter=%d eta_min=%f eta=%f eta_max=%f \n", iter, ls_iter, eta_min, eta, eta_max); 
    }
    if(alpha[i] <  DTAU)
		alpha[i]=0;
    if((C_i-alpha[i]) < DTAU)
		alpha[i]=C_i;
    if(alpha[j] <  DTAU)
		alpha[j]=0;
    if((C_j-alpha[j]) < DTAU)
		alpha[j]=C_j;  
    
		// update alpha_status Qalpha and Qalpha_bar_all
		{
			update_alpha_status(i);
			update_alpha_status(j);
			int k, n;
			if(ui != is_upper_bound(i))
      {
        // svnvish: BUGBUG
        // bad memory access pattern
        for(n=0;n<num_kernels;n++)
        {
          const Qfloat *Q_i=Q->get_Qn(n,i,l);
          if(ui)
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n] -= C_i * Q_i[k];
          else
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n]+=C_i * Q_i[k];
        }
      }
      
			if(uj != is_upper_bound(j))
      {
        // svnvish: BUGBUG
        // bad memory access pattern
        for(n=0;n<num_kernels;n++)
        {
          const Qfloat *Q_j=Q->get_Qn(n,j,l);
          if(uj)
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n] -= C_j * Q_j[k];
          else
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n]+=C_j * Q_j[k];
        }
      }
		}
    
    // Update G, G_bar, and QD
    g_from_Qalpha();
    QD=Q->get_QD();
    
    info("iter=%d, obj=%f \n", iter, obj);
  }
  
	// calculate rho
	si->rho=calculate_rho();

	// calculate objective value
	{
    si->obj=d_lp();
    for(int i=0;i<l;i++)
      si->obj+=(alpha[i]*p[i]); 
    // Remember we minimized the -ve of the dual?
    si->obj *= -1;
  }
  
	si->upper_bound_p=Cp;
	si->upper_bound_n=Cn;
	
  delete [] rr;

  info("\n LPQPSolver: optimization finished, #iter=%d\n",iter);
  return;
}

void Solver::EntSolver(SolutionInfo* si)
{
  // Find optimal d before getting into main loop 
  // Also compute initial objective function value   
  double alpha_sum=0.0;
  for(int i=0;i<l;i++)
    alpha_sum+=(alpha[i]*p[i]); 
  double obj=d_ent() + alpha_sum;
  info("initial obj=%f \n", -obj);
  
	// optimization step
	int iter=0;
	int counter=min(l,1000)+1;

	while(1)
	{    
		// show progress and do shrinking

		if(--counter == 0)
		{
			counter=min(l,1000);
			if(shrinking) do_shrinking();
			info(".");
		}

		int i,j;
    
		if(ent_select_working_set(i,j)!=0)
		{
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size=l;
      info("*");
      if(ent_select_working_set(i,j)!=0)
				break;
			else
				counter=1;	// do shrinking next iteration
		}
		
		++iter;

    double C_i=get_C(i);
    double C_j=get_C(j);
  
    bool ui=is_upper_bound(i);
    bool uj=is_upper_bound(j);

    double old_alpha_i=alpha[i];
    double old_alpha_j=alpha[j];
    
    // Find a direction of descent 
		double dir_i=0.0;
		double dir_j=0.0;

    double dir_grad=0.0;
    double dir_grad_offset=0.0; 
    double eta_max=0; // maximum step size 
    
    Qfloat *rr = new Qfloat[num_kernels]; // modified for windows
    
    if(y[i] == y[j])
    {
      // dir_i and dir_j must have opposite signs
      // First try dir_i=+1 and dir_j=-1
      // else try dir_i=-1 and dir_j=+1
      if(G[i] - G[j] < G[j] - G[i])
      {
        dir_i=1;
        dir_j=-1;
        dir_grad=G[i] - G[j];
        dir_grad_offset=p[i] - p[j];
        eta_max=min(C_i - alpha[i], alpha[j]);
      }
      else
      {
        dir_i=-1;
        dir_j=1;
        dir_grad=G[j] - G[i];
        dir_grad_offset=p[j] - p[i];
        eta_max=min(alpha[i], C_j - alpha[j]);
      }
      for(int n=0;n<num_kernels;n++)
      {
        const Qfloat *Q_i=Q->get_Qn(n, i, active_size);
        const Qfloat *Q_j=Q->get_Qn(n, j, active_size);
        rr[n]=Q_i[i] + Q_j[j] - 2*Q_i[j];
      }
    }
    else
    {
      // dir_i and dir_j must have same sign
      // First try dir_i=+1 and dir_j=+1
      // else try dir_i=-1 and dir_j=-1
      if(G[i] + G[j] < -G[j] - G[i])
      {
        dir_i=1;
        dir_j=1;
        dir_grad=G[i] + G[j];
        dir_grad_offset=p[i] + p[j];
        eta_max=min(C_i - alpha[i], C_j - alpha[j]);
      }
      else
      {
        dir_i=-1;
        dir_j=-1;
        dir_grad=-G[i] -G[j];
        dir_grad_offset=-p[i] -p[j];
        eta_max=min(alpha[i], alpha[j]);
      }
      for(int n=0;n<num_kernels;n++)
      {
        const Qfloat *Q_i=Q->get_Qn(n, i, active_size);
        const Qfloat *Q_j=Q->get_Qn(n, j, active_size);
        rr[n]=Q_i[i] + Q_j[j] + 2*Q_i[j];
      }
    }
    
    assert(dir_grad < 0);
    assert(eta_max > 0);
    
    // Compute directional Hessian
    double dir_hess=-(dir_grad-dir_grad_offset)*(dir_grad-dir_grad_offset)/lambda;
    for(int n=0;n<num_kernels;n++)
      if(d[n] > DTAU)
      {
        double g_sq=dir_i*Qalpha_all[i][n] + dir_j*Qalpha_all[j][n];
        dir_hess+=d[n]*(rr[n] + (g_sq*g_sq/lambda)); 
      }
    if(dir_hess <= 0)
      dir_hess=TAU;
    // assert(dir_hess > 0);
    double old_obj=obj;    
    double old_dir_grad=dir_grad;
    double eta_min=0.0;
    double eta=-dir_grad/dir_hess;
    if(eta > eta_max)
      eta=eta_max;
    if(eta < eta_min)
      eta=eta_min + 0.5*(eta_max - eta_min);
    int ls_iter=0;
    
    while(1)
    {
      double new_alpha_i=old_alpha_i + eta*dir_i;
      double new_alpha_j=old_alpha_j + eta*dir_j;
      double change_i=new_alpha_i - alpha[i];
      double change_j=new_alpha_j - alpha[j];      
      alpha[i]=new_alpha_i;
      alpha[j]=new_alpha_j;
      update_alpha_status(i);
      update_alpha_status(j);
      update_Qalpha(i, j, change_i, change_j);
      alpha_sum+=(p[i]*change_i + p[j]*change_j); 
      obj=d_ent() + alpha_sum;      
      // Compute new directional gradient
      dir_grad=dir_grad_offset;
      for(int n=0;n<num_kernels;n++)
        if(d[n] > DTAU)
          dir_grad+=d[n]*(dir_i*Qalpha_all[i][n] + dir_j*Qalpha_all[j][n]);
      
      if((obj < old_obj + obj_threshold*eta*old_dir_grad) ||
         (fabs(eta_max - eta_min) < diff_threshold) ||
         (fabs(dir_grad) < diff_threshold) ||
         (eta == eta_max && dir_grad < 0))
      {
        if((obj > old_obj) || (eta <= 0))
        {
          info("\n iter=%d ls_iter=%d \n", iter, ls_iter);
          info("eta=%f eta_max=%f eta_min=%f \n", eta, eta_max, eta_min);
          info("old_obj=%f obj=%f old_dir_grad=%f dir_grad=%f \n", old_obj, obj, old_dir_grad, dir_grad);
        }
        break;
      }
			
      if(dir_grad > 0)
        eta_max=eta;
      else
      {
        if(obj > old_obj)
        {
          info("iter=%d ls_iter=%d \n", iter, ls_iter);
          info("eta=%f eta_max=%f eta_min=%f \n", eta, eta_max, eta_min);
          info("old_obj=%f obj=%f old_dir_grad=%f dir_grad=%f \n", old_obj, obj, old_dir_grad, dir_grad);
        }
        if(eta >= eta_min+TAU)
          assert(obj <= old_obj);
        
        eta_min=eta;
      }
      
      // Try a Newton-Raphson step
      // Compute directional Hessian
      dir_hess=-(dir_grad-dir_grad_offset)*(dir_grad-dir_grad_offset)/lambda;
      for(int n=0;n<num_kernels;n++)
        if(d[n] > DTAU)
        {
          double g_sq=dir_i*Qalpha_all[i][n] + dir_j*Qalpha_all[j][n];
          dir_hess+=d[n]*(rr[n] + (g_sq*g_sq/lambda)); 
        }
      if(dir_hess <= 0)
        dir_hess=TAU;
      // assert(dir_hess > 0);
      eta=eta - dir_grad/dir_hess;
      // If the result is outside the brackets, use bisection
      if(eta > eta_max || eta <= eta_min)
        eta=eta_min + 0.5*(eta_max - eta_min);
      
      ls_iter++;
			
      if(ls_iter > 20)
        info("iter=%d ls_iter=%d eta_min=%f eta=%f eta_max=%f \n", iter, ls_iter, eta_min, eta, eta_max); 
    }
    
    
    // update alpha_status and Qalpha_bar_all
		{
			update_alpha_status(i);
			update_alpha_status(j);
			int k, n;
			if(ui != is_upper_bound(i))
      {
        // svnvish: BUGBUG
        // bad memory access pattern
        for(n=0;n<num_kernels;n++)
        {
          const Qfloat *Q_i=Q->get_Qn(n,i,l);
          if(ui)
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n] -= C_i * Q_i[k];
          else
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n]+=C_i * Q_i[k];
        }
      }
      
			if(uj != is_upper_bound(j))
      {
        // svnvish: BUGBUG
        // bad memory access pattern
        for(n=0;n<num_kernels;n++)
        {
          const Qfloat *Q_j=Q->get_Qn(n,j,l);
          if(uj)
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n] -= C_j * Q_j[k];
          else
            for(k=0;k<l;k++)
              Qalpha_bar_all[k][n]+=C_j * Q_j[k];
        }
      }
		}
    
    // Update G and G_bar and QD
    g_from_Qalpha();
    QD=Q->get_QD();

    info("EntSolver: iter=%d obj=%f \n", iter, obj);
   delete [] rr;	

  }
  
	// calculate rho
	si->rho=calculate_rho();

	// calculate objective value
	{
    si->obj=d_ent();
    for(int i=0;i<l;i++)
      si->obj+=(alpha[i]*p[i]); 
    // Remember we minimized the -ve of the dual?
    si->obj *= -1;
  }
  
	si->upper_bound_p=Cp;
	si->upper_bound_n=Cn;
  info("\n EntSolver: optimization finished, #iter=%d\n",iter);
  return;
  
}

// Copy solution, report stats etc
void Solver::finalize(const double* dk,
                      const double& obj, 
                      SolutionInfo* si)
{
  // Copy solution 
  for(int n=0;n<num_kernels;n++)
    d[n]=dk[n];
  
  si->rho=calculate_rho();
  si->obj=obj; 
  si->upper_bound_p=Cp;
  si->upper_bound_n=Cn;
  
  info("\n \n");
  info("SVM calls: %d \n", fun_timer.num_calls);
  info("CPU time spent in SVM calls: %f \n", fun_timer.total_cpu);
  info("\n");
  fun_timer.reset();
  delete [] dk;	 // modified for windows
  return;
}

double norm_l2(double *x, int dim)
{
  double norm_l2=0.0;
  for(int n=0;n<dim;n++)
    norm_l2+=x[n]*x[n];
  return sqrt(norm_l2);
}
  
double norm_inf(double *x, int dim)
{
  double norm_inf=0.0;
  for(int n=0;n<dim;n++)
  {
    double tmp=fabs(x[n]);
    if( tmp > norm_inf)
      norm_inf=tmp;
  }
  return norm_inf;
}

// Normalize vector to be of unit norm 
void Solver::normalize(double* gradk)
{
  double grad_norm_l2=norm_l2(gradk, num_kernels);
  for(int n=0;n<num_kernels;n++)
    gradk[n] /= grad_norm_l2;
  return;
}

// Check for convergence 
bool Solver::converged(const double* dk, 
                       double* gradk, 
                       const double primal,
                       obj_grad_c *obj_grad_fp,
											 proj_c *proj_fp)
{
  
  double *proj_grad = new double[num_kernels];
  
  // Project along negative gradient direction to check for convergence 
  for(int n=0;n<num_kernels;n++)
      proj_grad[n]=dk[n] - gradk[n];
  proj_fp->proj(proj_grad);

  // proj_grad now contains projected values
  // subtract dk from it
  for(int n=0;n<num_kernels;n++)
    proj_grad[n] -= dk[n];
  
  double grad_norm_l2=norm_l2(gradk, num_kernels);
  double pg_norm_l2=norm_l2(proj_grad, num_kernels);
  double pg_norm_inf=norm_inf(proj_grad, num_kernels);
  double dual=dual_obj(obj_grad_fp);
  double gap=fabs(primal - dual);

  info("g_L2=%f pg_L2=%f Inf=%f Dual=%f gap=%f eps=%f", grad_norm_l2, pg_norm_l2, pg_norm_inf, dual, gap, eps);
  delete [] proj_grad;	
  bool result_final = (pg_norm_inf < ProjGrad::inf_abs_tol || pg_norm_l2 < ProjGrad::l2_abs_tol || gap < ProjGrad::kkt_gap_tol);
if(!result_final){	
  if(tune_precision == 1){
	if(pg_norm_l2 <= 5 && pg_norm_l2 >= 1 && f1){
		eps = 0.01;
		f1=false;
		obj_true = primal_obj(false, obj_grad_fp);
		grad(gradk, obj_grad_fp);
		}
	else if(pg_norm_l2 < 1 && f2){
		eps = 0.001;
		f2=false;f1=false;
		obj_true = primal_obj(false, obj_grad_fp);
		grad(gradk, obj_grad_fp);
		}
		
  }
}				
return(result_final);

}

void Solver::ReducedGradient(SolutionInfo* si, 
                             obj_grad_c *obj_grad_fp,
														 proj_c *proj_fp)
{

  fun_timer.start();

  double gs_delta_max=ReducedGradient::gs_delta_init;
  
  // k-th iterate
  double *dk = new double[num_kernels]; // modified for windows
  
  // k-th descent direction 
  double *dirk = new double[num_kernels]; // modified for windows
  for(int n=0;n<num_kernels;n++)
  {
    dk[n]=d[n];
    dirk[n]=0;
  }
  
  // Compute objective function value and gradient at d_{0}
  double obj=primal_obj(true, obj_grad_fp);

  double *gradk = new double[num_kernels]; // modified for windows 
  grad(gradk, obj_grad_fp);

  info("init obj=%f \n", obj);
  
  for(int iter=1; iter <= ProjGrad::max_iter; iter++)
  {
    info("%.3d ", iter);
    
    if(converged(dk, gradk, obj, obj_grad_fp, proj_fp)){
		delete [] dirk; // modified for windows
		delete [] gradk; // modified for windows
		return finalize(dk, obj, si);
	}
    normalize(gradk);
    
    // Find index of maximum d value 
    int d_max_idx=0;
    double d_max=dk[0];
    for(int n=0;n<num_kernels;n++)
      if(dk[n] > d_max)
      {
        d_max=dk[n];
        d_max_idx=n;
      }
    proj_fp->dd(dk, gradk, dirk, d_max_idx);

    // Initial loop
    // Take max step along the reduced gradient direction
    
    double step_max=INF;
    double step_min=0;
    
    double obj_max=0;
    double obj_min=obj;

    for(int n=0;n<num_kernels;n++)
      if(dirk[n] < 0 && step_max > (-dk[n]/dirk[n]))
        step_max=(-dk[n]/dirk[n]);
    
    step_max=min(step_max, ReducedGradient::step_clip);
    assert(step_max > 0);
    
    double delta_max=step_max;
        
    while(obj_max < obj_min)
    {
      for(int n=0;n<num_kernels;n++)
        d[n]=dk[n] + step_max*dirk[n];
      
      obj_max=primal_obj(false, obj_grad_fp); 
            
      if(obj_max < obj_min)
      {
        obj_min=obj_max;
        for(int n=0;n<num_kernels;n++)
          dk[n]=d[n];
        
        proj_fp->dd(dk, gradk, dirk, d_max_idx);
        
        step_max=INF;
        for(int n=0;n<num_kernels;n++)
          if(dirk[n] < 0 && step_max > (-dk[n]/dirk[n]))
            step_max=(-dk[n]/dirk[n]);
        if(step_max == INF)
        {
          step_max=0;
          delta_max=0;
        }
        else
        {
          delta_max=step_max;
          obj_max=0;
        }
      }
    }
    
    // Line search 
    double step=0.0;
    
    if(obj_min < obj_max)
    {
      obj=obj_min;
      step=step_min;
    }
    else
    {
      obj=obj_max;
      step=step_max;
    }
    
    while((step_max - step_min)>gs_delta_max*fabs(delta_max) && 
          step_max > DBL_EPSILON)
    {
      double step_r=step_min + (step_max - step_min)/ReducedGradient::gold;
      for(int n=0;n<num_kernels;n++)
        d[n]=dk[n] + step_r*dirk[n];
      double obj_r=primal_obj(false, obj_grad_fp); 
      
      double step_l=step_min + (step_r - step_min)/ReducedGradient::gold;
      for(int n=0;n<num_kernels;n++)
        d[n]=dk[n] + step_l*dirk[n];
      double obj_l=primal_obj(false, obj_grad_fp); 
      
      if((obj_min < obj_l) &&
         (obj_min < obj_r) &&
         (obj_min < obj_max)){
        
        obj=obj_min;
        step=step_min;
        step_max=step_l;
        obj_max=obj_l;
      
      }else if ((obj_l < obj_min) &&
                (obj_l < obj_r) &&
                (obj_l < obj_max))
      {
       
        obj=obj_l;
        step=step_l;
        step_max=step_r;
        obj_max=obj_r;
        
        
      }else if ((obj_r < obj_min) &&
                (obj_r < obj_l) &&
                (obj_r < obj_max))
      {
       
        obj=obj_r;
        step=step_r;
        step_min=step_l;
        obj_min=obj_l;
        
      }else if ((obj_max < obj_min) &&
                (obj_max < obj_l) &&
                (obj_max < obj_r))
      {
        
        obj=obj_max;
        step=step_max;
        step_min=step_r;
        obj_min=obj_r;
        
      }

    }

    info("obj value=%f \n", obj);

    // Update and compute gradients
    for(int n=0;n<num_kernels;n++)
    {
      dk[n]+=step*dirk[n];
      d[n]=dk[n];
    }
    grad(gradk, obj_grad_fp);
  }
  
  si->obj=-1;
  si->rho=0;
  si->upper_bound_p=Cp;
	si->upper_bound_n=Cn;

  delete [] dk;										// modified for windows
  delete [] gradk;									// modified for windows
  delete [] dirk;									// modified for windows
    
  info("Failure in Reduced Gradient optimizer \n");
  return;  
}

void Solver::MirrorDescent(SolutionInfo* si, 
                           obj_grad_c *obj_grad_fp,
													 proj_c *proj_fp)
{
  
  fun_timer.reset();
  
  // k-th iterate
  double *dk = new double[num_kernels];				// modified for windows
  // k-th gradient 
  double *gradk = new double[num_kernels]; 			// modified for windows
  // k-th descent direction
  double *dirk = new double[num_kernels];			// modified for windows
  for(int n=0;n<num_kernels;n++)
    dk[n]=d[n];
  proj_fp->proj(dk);
  
  // Compute f and its gradient at x_{0}
  double obj=primal_obj(true, obj_grad_fp);
  grad(gradk, obj_grad_fp);

  info("init obj=%f \n", obj);
  
  double step0=sqrt(log(double(num_kernels)));		// modified for windows
  
  for(int iter=1; iter <= ProjGrad::max_iter; iter++)
  {
    info("%.3d ", iter);
        
    if(converged(dk, gradk, obj, obj_grad_fp, proj_fp)){
      delete [] gradk; // modified for windows
	  delete [] dirk; // modified for windows
	  return finalize(dk, obj, si);
  }

    // Step 2: tune step size 

    // Step 2.1: Compute infinity norm of gradient 
    double grad_norm_inf=norm_inf(gradk, num_kernels); 
    
    // Step 2.2: Actual step computation 
    double step=step0/(sqrt((double)iter)*grad_norm_inf);
    
    // Step 3: Compute descent direction 
    for(int n=0;n<num_kernels;n++)
      dirk[n]=-step*gradk[n] + log(d[n]) + 1;
    
    // Step 4
    // Perform EG update along dir_{k}
    
    // safe exponentiation 
    double exp_max=-INF;
    for(int n=0;n<num_kernels;n++)
      if(dirk[n] > exp_max) exp_max=dirk[n];
    
    double dsum=0.0;
    for(int n=0;n<num_kernels;n++)
    {
      d[n]=exp(dirk[n] - exp_max);
      dsum+=d[n];
    }
    for(int n=0;n<num_kernels;n++)
    {
      d[n] /= dsum;
      dk[n]=d[n];
    }
    
    obj=primal_obj(false, obj_grad_fp); 
    grad(gradk, obj_grad_fp);
    info("Obj=%f \n", obj);
  }
  si->obj=-1;
  si->rho=0;
  si->upper_bound_p=Cp;
	si->upper_bound_n=Cn;
  
  delete [] dk;										// modified for windows
  delete [] gradk;									// modified for windows
  delete [] dirk;									// modified for windows
  
  info("Failure in mirror descent optimizer \n");
  return;  
}


// SPG-GMKL

void Solver::SPGSolver(SolutionInfo* si, 
                       obj_grad_c *obj_grad_fp,
											 proj_c *proj_fp)
{
	f1=true;f2=true;
	svm_calls = 0;
	if(tune_precision == 1)
		eps = 0.1;
		
  fun_timer.reset();
  // k-th iterate

  double *dk = new double[num_kernels];
  double *d_temp = new double[num_kernels];	
  // k-th descent direction 
  double *dirk = new double[num_kernels];
  double *proj_dirk = new double[num_kernels];

  for(int n=0;n<num_kernels;n++)
  {
    dk[n]=d[n];
    dirk[n]=0.0;
  }
  proj_fp->proj(dk);
  
  // Compute f and its gradient at x_{0}
  // Push into fk array
  double obj=primal_obj(true, obj_grad_fp);
  double *gradk = new double[num_kernels];
  grad(gradk, obj_grad_fp);

  double qk=0.0;
  double ck=0.0;
  double etak=ProjGrad::loose_hz;

  double sksk=0.0;
  double skyk=0.0;
info("###\n");
        solver_timer.stop();    
  info("init Obj=%f time=%f \n", obj,solver_timer.total_cpu);
solver_timer.start();

//  info("init obj=%f \n", obj);
 // info("###\n");
   my_alpha=1.0/norm_inf(gradk, num_kernels);
  for(int iter=1; iter <= ProjGrad::max_iter; iter++)
  {
    info("%.3d ", iter);
    obj_true = obj;
    if(converged(dk, gradk, obj, obj_grad_fp, proj_fp)) {
        solver_timer.stop();
        info(" Obj=%f time=%f \n", obj,solver_timer.total_cpu);
        solver_timer.start();
                info("###\n");
           printf("\nSolver Converged\n");
      return finalize(dk, obj, si);

  }
	obj = obj_true;
    // Step 2: Backtracking
    // proposed stepsize 
    double step=1.0;

    // Step 2.1: Compute dir_{k}=P(d_{k} - \alpha_{k} g_{k})

		
if(use_SPG == 1){
    for(int n=0;n<num_kernels;n++){
        dirk[n]=dk[n] - my_alpha*gradk[n];
		proj_dirk[n]=dk[n] - gradk[n];
	}
    proj_fp->proj(dirk);
    proj_fp->proj(proj_dirk);
    // Compute dtg=\inner{dir_{k}}{g_{k}} 
    double dtg=0.0;
    double dkdk=0.0;
    double pg = 0.0;
    for(int n=0;n<num_kernels;n++)
    {
		dirk[n]-=dk[n];
		proj_dirk[n]-=dk[n];
		pg += proj_dirk[n]*proj_dirk[n];
      dtg+=dirk[n]*gradk[n];
      dkdk+=dirk[n]*dirk[n];
    }
    // info("dtg=%.5f dkdk=%.5f ", dtg,dkdk);
   
    if(monotone == 1)
		etak = 0;
		
    //info("etak=%.6f ", etak);
    double newqk=etak*qk+1.0;
    ck=(etak*qk*ck+obj)/newqk;
    qk=newqk;
    bool flag=true;
    double rho=0.0;
    svm_periter = 0;
    while(true)
    {

      // Step 2.2: Set d_{+}=d_{k} + \step dir_{k}
      for(int n=0;n<num_kernels;n++)
        d[n]=dk[n] + step*dirk[n];

      double objplus=primal_obj(false, obj_grad_fp); 
     

		  
      if(flag)
      {
        flag=false;
        double pred_decr=-dtg-dkdk/(2.0*my_alpha);
        double actual_decr=obj-objplus;
        if(pred_decr)
          rho=actual_decr/pred_decr;
        else
          rho = ProjGrad::good_ratio+1; 
        //info("rho=%.6f ", rho);
      }
      
      // Step 2.3 
      // if f(d_{+}) \leq obj_max + gamma * \inner{dir_{k}}{g_{k}}}
      if((step < ProjGrad::min_step) || 
         (objplus<=(ck + ProjGrad::gamma*step*dtg)))
      {

		//  if(sqrt(pg) < 0.01 && step < 1e-8 && !xyz){
		//	xyz = false;
		//	eps = eps * 0.1;
		 // }
        // Success in step 2.3
        // d_{k+1}=d_{+} 
        // s_{k}=d_{k+1} - d_{k}=step*dir_{k} 
        // y_{k}=g_{k+1} - g_{k} 
     
      if(use_SPG == 1){
		  if(step < 1e-8){
			etak=max(ProjGrad::tight_hz, etak-0.025);
		  }
			
        if(step < 1e-8 && eps > 1e-5){
			eps = eps * 0.1;
			if(eps == 0.01)
				f1 = false;
			else if(eps <= 0.001){
				f1 = false;f2 = false;
			}
			for(int n=0;n<num_kernels;n++){
				d_temp[n] = d[n];
				d[n] = dk[n];
			}
			obj = primal_obj(false, obj_grad_fp);
			  grad(gradk, obj_grad_fp);
			for(int n=0;n<num_kernels;n++)
				dirk[n]=dk[n] - my_alpha*gradk[n];
		    proj_fp->proj(dirk);
		    dtg=0.0;dkdk=0.0;
		    for(int n=0;n<num_kernels;n++){
				dirk[n]-=dk[n];
				dtg+=dirk[n]*gradk[n];
				dkdk+=dirk[n]*dirk[n];
				d[n] = d_temp[n];
			}
			objplus=primal_obj(false, obj_grad_fp);
		}	
	}
        
        double *gradplus = new double[num_kernels];
        grad(gradplus, obj_grad_fp);
         info(" step=%.6f svm=%d ",step,svm_periter);       
        sksk=0.0; 
        skyk=0.0;
        for(int n=0;n<num_kernels;n++)
        {
          sksk+=dirk[n]*dirk[n]; 
          skyk+=dirk[n]*(gradplus[n] - gradk[n]);
          // adjust iterate and gradient values for next time
          dk[n]=d[n];
          gradk[n]=gradplus[n];
        }
        // Take into account the stepsize 
        // sk = x_{t+1} - x_{t} = step*dirk
        sksk*=(step*step);
        skyk*=step;
        
        if(skyk <= 0.0)
        {
          // Step in a direction of negative curvature. This should
          // be large enough to allow good progress.
          double grad_norm_inf=norm_inf(gradk, num_kernels);
          my_alpha=1.0/grad_norm_inf; 
        }
        else
          my_alpha=min(ProjGrad::alpha_max, max(ProjGrad::alpha_min, sksk/skyk));
          
        // Store function value 
			obj=objplus;
          
        if(rho < ProjGrad::bad_ratio)
          etak=max(ProjGrad::tight_hz, etak-0.025);
        else if(rho > ProjGrad::good_ratio)
          etak=min(ProjGrad::loose_hz, etak+0.025);
        
        solver_timer.stop();
        info("Obj=%f time=%f \n", objplus,solver_timer.total_cpu);
        solver_timer.start();
     
 //  info("Obj=%f \n", objplus);
        break;
        
      }else{
        
        // Failure in step 2.3
        // Compute a safeguarded new trial steplength
        if(step > 0.1)
        {
          double stepnew=-0.5*step*dtg/(objplus-obj-step*dtg);
          if(stepnew >= 0.9)
            stepnew=ProjGrad::decay;
          double steptry=step*stepnew;
          step=steptry > 0.1 ? steptry:step*ProjGrad::decay;
        }
        else
          step *= ProjGrad::decay;
        continue;
      }
    }
  }
else{
	for(int n=0;n<num_kernels;n++){
		proj_dirk[n]=dk[n] - gradk[n];
	}
    proj_fp->proj(proj_dirk);
    // Compute dtg=\inner{dir_{k}}{g_{k}} 
    double pg = 0.0;
    for(int n=0;n<num_kernels;n++)
    {
		proj_dirk[n]-=dk[n];
		pg += proj_dirk[n]*proj_dirk[n];
    }
    svm_periter = 0;
      while(true)
    {

      for(int n=0;n<num_kernels;n++)
        d[n]=dk[n] - step*gradk[n];
		proj_fp->proj(d);

      double objplus=primal_obj(false, obj_grad_fp); 

      if((step < ProjGrad::min_step) || 
         (objplus<=(obj - ProjGrad::gamma*step*pg)))
      {

        double *gradplus = new double[num_kernels];
        grad(gradplus, obj_grad_fp);
         info(" step=%.6f svm=%d ",step,svm_periter);       
        for(int n=0;n<num_kernels;n++)
        {
          dk[n]=d[n];
          gradk[n]=gradplus[n];
        }
			obj=objplus;
        solver_timer.stop();
        info("Obj=%f time=%f \n", objplus,solver_timer.total_cpu);
        solver_timer.start();

//		info("Obj=%f \n", objplus);
        break;
        
      }
      else{
        
        // Failure in step 2.3
        // Compute a safeguarded new trial steplength
        if(step > 0.1)
        {
          double stepnew=0.5*step*pg/(objplus-obj+step*pg);
          if(stepnew >= 0.9)
            stepnew=ProjGrad::decay;
          double steptry=step*stepnew;
          step=steptry > 0.1 ? steptry:step*ProjGrad::decay;
        }
        else
          step *= ProjGrad::decay;
        continue;
      }
    }
	
}	  
}
  si->obj=-1;
  si->rho=0;
  si->upper_bound_p=Cp;
	si->upper_bound_n=Cn;
	
  delete [] dk;	
  delete [] gradk;
  delete [] dirk;	
  
  info("Failure in projected gradient optimizer \n");
  return finalize(dk, obj, si);  
}


void Solver::swap_index(int i, int j)
{
	Q->swap_index(i,j);
	swap(y[i],y[j]);
	swap(G[i],G[j]);
	swap(Qalpha_all[i],Qalpha_all[j]);
	swap(alpha_status[i],alpha_status[j]);
	swap(alpha[i],alpha[j]);
	swap(p[i],p[j]);
	swap(active_set[i],active_set[j]);
	swap(G_bar[i],G_bar[j]);
	swap(Qalpha_bar_all[i],Qalpha_bar_all[j]);
}

void Solver::reconstruct_gradient()
{
  
	// reconstruct inactive elements of G from G_bar and free variables
	if(active_size == l) return;

  g_from_Qalpha();
  
	int i,j;
	int nr_free=0;

	for(j=active_size;j<l;j++)
		G[j]=G_bar[j] + p[j];

	for(j=0;j<active_size;j++)
		if(is_free(j))
			nr_free++;

	if(2*nr_free < active_size)
		info("\nWarning: using -h 0 may be faster nr_free=%d active_size=%d l=%d\n", nr_free, active_size, l);
  
	if (nr_free*l > 2*active_size*(l-active_size))
	{
		for(i=active_size;i<l;i++)
		{
			const Qfloat *Q_i=Q->get_Q(i,active_size);
			for(j=0;j<active_size;j++)
				if(is_free(j))
					G[i]+=alpha[j] * Q_i[j];
		}
	}
	else
	{
		for(i=0;i<active_size;i++)
			if(is_free(i))
			{
				const Qfloat *Q_i=Q->get_Q(i,l);
				double alpha_i=alpha[i];
				for(j=active_size;j<l;j++)
					G[j]+=alpha_i * Q_i[j];
			}
	}

}

// initialize Qalpha and Qalpha_bar for all kernels
// Also compute and store grad_d 
void Solver::init_Qalpha(void)
{  
  int i, n;
  for(i=0;i<l;i++)
    for(n=0;n<num_kernels;n++)
    {
      Qalpha_all[i][n]=0.0;
      Qalpha_bar_all[i][n]=0.0;
    }
  
  
  for(n=0;n<num_kernels;n++)
    grad_d[n]=0.0;
  
  for(i=0;i<l;i++)
  {
    if(!is_lower_bound(i))
    {
      double alpha_i=alpha[i];
      for(n=0;n<num_kernels;n++)
      {
        const Qfloat *Q_i=Q->get_Qn(n,i,l);
        int j;
        for(j=0;j<l;j++)
        {
          double tmp=alpha_i*Q_i[j];
          Qalpha_all[j][n]+=tmp;
          grad_d[n]+=tmp*alpha[j];
        }
        if(is_upper_bound(i))
        {
          for(j=0;j<l;j++)
            Qalpha_bar_all[j][n]+=get_C(i)*Q_i[j];
        }
      }
    }
  }
  for(n=0;n<num_kernels;n++)
  {
    grad_d[n] *= -0.5; 
    if(grad_d[n] > 0)
    {
      // info("BUGBUG: init_Qalpha: grad_d[%d]=%f \n", n, grad_d[n]);
      grad_d[n]=0.0;
    }
  }
}

// Also update grad_d 
void Solver::update_Qalpha(const int& i, 
                           const int& j, 
                           const double& delta_alpha_i,
                           const double& delta_alpha_j)
{
  int n, k;
  for(n=0;n<num_kernels;n++)
  {
    const Qfloat *Q_i=Q->get_Qn(n,i,active_size);
    const Qfloat *Q_j=Q->get_Qn(n, j,active_size);
    
    // update gradient w.r.t. d 
    // 0.5*(delta_alpha^T Q delta_alpha)
    grad_d[n] -= (0.5*Q_i[i]*delta_alpha_i*delta_alpha_i + 
                  0.5*Q_j[j]*delta_alpha_j*delta_alpha_j + 
                  Q_i[j]*delta_alpha_i*delta_alpha_j);
    // delta_alpha^T Q alpha
    grad_d[n] -= (Qalpha_all[i][n]*delta_alpha_i + 
                  Qalpha_all[j][n]*delta_alpha_j); 

    for(k=0;k<active_size;k++)
      Qalpha_all[k][n]+=Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
  }
  
}

void Solver::g_from_Qalpha(void)
{
  int i, n;

  for(i=0;i<l;i++)
  {
    G[i]=p[i];
    G_bar[i]=0;
    for(n=0;n<num_kernels;n++)
    {
      if(d[n] > DTAU)
      {
        G[i]+=d[n]*Qalpha_all[i][n];
        G_bar[i]+=d[n]*Qalpha_bar_all[i][n];
      }
    }
  }
}

double Solver::d_l2(void)
{
  double obj=0.0;
  for(int n=0;n<num_kernels;n++){
    d[n]=-grad_d[n]/lambda;
    obj+=grad_d[n]*grad_d[n];
  }
  return obj/(2.0*lambda);
}

double Solver::d_lp(void)
{
/* When L_p is small (say 1.01) then L_q is large (approx 100)
 * If \grad_d[n] is large then \grad_d[n]^L_q may overflow and go to infinity
 * */

  double obj=0.0;
  double gmax=-INF;
 for(int n=0;n<num_kernels;n++)
    if(-grad_d[n] > gmax) gmax=-grad_d[n];
    
  for(int n=0;n<num_kernels;n++)
    if(-grad_d[n]>DTAU)
      obj+=pow((-grad_d[n]/gmax), (double) L_q);
  
  double mult=0.0;
  if(obj>0) //DTAU
  {
    mult=pow(obj, (2.0 - L_q)/L_q)/lambda;
    obj=gmax*gmax*pow(obj, 2.0/L_q)/(2*lambda);
  }
  for(int n=0;n<num_kernels;n++)
  {
    if(-grad_d[n]>DTAU)
      d[n]=pow((-grad_d[n]/gmax), (double) (L_q-1))*mult*gmax;
    else
      d[n]=0.0;
  }

  return obj;
}

double Solver::d_ent(void)
{
  // Find max element 
  double exp_max=-INF; 
  int n;
  double dsum=0.0;
  for(n=0;n<num_kernels;n++)
    if(-grad_d[n] > exp_max) exp_max=-grad_d[n];
  
  // safe exponentiation 
  for(n=0;n<num_kernels;n++)
  {
    d[n]=exp((-grad_d[n] - exp_max)/lambda);
    dsum+=d[n];
  }
  for(n=0;n<num_kernels;n++)
    d[n] /= dsum;
  return exp_max + (log(dsum)*lambda);
}

double Solver::primal_obj(bool reinit, obj_grad_c *obj_grad_fp)
{
  fun_timer.start();
	svm_calls++;
	svm_periter++;
  QD=Q->get_QD();  
  
  if(reinit)
  {
    for(int i=0; i < l; i++)
    {
      alpha[i]=0.0;
      update_alpha_status(i);
    }
    active_size=l;
    init_Qalpha();
  }
  g_from_Qalpha();
  SolutionInfo si;
  QPSolver(&si);
	double obj= -si.obj;
  if(lambda)
    obj+=lambda*obj_grad_fp->primal();
  fun_timer.stop();
  return obj; 
}

double Solver::dual_obj(obj_grad_c *obj_grad_fp)
{
  double dual_obj=0.0;
  
  for(int i=0;i<l;i++)
    dual_obj -= (alpha[i]*p[i]); 
  if(lambda)
    dual_obj -= lambda*obj_grad_fp->dual();
  
  return dual_obj; 
}

double *Solver::grad(double *gradk, obj_grad_c *obj_grad_fp)
{
  g_from_Qalpha();  
  if(lambda)
  {
    obj_grad_fp->grad(gradk, lambda);
    for(int n=0;n<num_kernels;n++)
      gradk[n]+=grad_d[n];
  }
  else
    for(int n=0;n<num_kernels;n++)
      gradk[n]=grad_d[n];
  
  return gradk;
} 


// Compute solution to the Dai Fletcher projection problem
//
// min_x 0.5*x'*x - x'*z - lambda*(a'*x - b)
// s.t. l \leq x \leq u
//
// Return the optimal value in x

double phi(double* x, 
           const double* a, 
           const double& b, 
           const double* z, 
           const double* l, 
           const double* u,
           const double& lambda,
           const int& n){
  double r=-b;
  
  for (int i=0;i<n;i++){
    x[i]=z[i] + lambda*a[i];
    if (x[i] > u[i]) 
      x[i]=u[i];
    else if(x[i] < l[i]) 
      x[i]=l[i];
    r+=a[i]*x[i];
  }
  return r;
}

// Dai-Fletcher Algorithm 1 (special case):
//
// Compute solution to the following QP
// 
// min_x  0.5*x'*x - x'*z
// s.t.   a'*x=b
//        l \leq x \leq u  
//

int project(double* x,
            const double* a, 
            const double& b, 
            const double* z, 
            const double* l, 
            const double* u, 
            const int& max_iter,
            const int& n){
  
  double r, r_l, r_u, s;
  double d_lambda=0.5, lambda=0.0;
  double lambda_l, lambda_u, lambda_new; 

  int inner_iter=1;
  
  // Bracketing 
  r=phi(x, a, b, z, l, u, lambda, n);
  
  if (r < 0){
    lambda_l=lambda;
    r_l=r;
    lambda+=d_lambda;
    r=phi(x, a, b, z, l, u, lambda, n);
    while(r < 0 && d_lambda < INF){
      lambda_l=lambda;
      s=max((r_l/r) - 1.0, 0.1);
      d_lambda+=(d_lambda/s);
      lambda+=d_lambda;
      r_l=r;
      r=phi(x, a, b, z, l, u, lambda, n);
    }
    lambda_u=lambda;
    r_u=r;
  }else{
    lambda_u=lambda;
    r_u=r;
    lambda -= d_lambda;
    r=phi(x, a, b, z, l, u, lambda, n);
    while(r > 0 && d_lambda > -INF){
      lambda_u=lambda;
      s=max((r_u/r) - 1.0, 0.1);
      d_lambda+=(d_lambda/s);
      lambda -= d_lambda;
      r_u=r;
      r=phi(x, a, b, z, l, u, lambda, n);
    }
    lambda_l=lambda;
    r_l=r;
  }

  if(fabs(d_lambda) > INF) {
    info("ERROR: Detected Infeasible QP! \n");
    return -1;
  }
  
  if(r_u == 0){
    lambda=lambda_u;
    r=phi(x, a, b, z, l, u, lambda, n);
    return (int) inner_iter;
  }

  // Secant phase 
  
  s=1.0 - (r_l/r_u);
  d_lambda=d_lambda/s;
  lambda=lambda_u - d_lambda;
  r=phi(x, a, b, z, l, u, lambda, n);
  
  while((fabs(r) > DaiFletcher::tol_r) && 
        (d_lambda > DaiFletcher::tol_lam * (1.0 + fabs(lambda)))
        && inner_iter < max_iter ){
    
    inner_iter++;
    if(r > 0){
      if(s <= 2.0){
        lambda_u=lambda;
        r_u=r;
        s=1.0 - r_l/r_u;
        d_lambda=(lambda_u - lambda_l)/s;
        lambda=lambda_u - d_lambda;
      }else{
        s=max(r_u/r - 1.0,0.1);
        d_lambda=(lambda_u - lambda)/s;
        lambda_new=max(lambda - d_lambda, 
                              0.75*lambda_l + 0.25*lambda);
        lambda_u=lambda;
        r_u=r;
        lambda= lambda_new;
        s=(lambda_u - lambda_l)/(lambda_u - lambda);
      }
    }else{
      if(s >= 2.0){
        lambda_l=lambda;
        r_l=r;
        s=1.0 - r_l/r_u;
        d_lambda=(lambda_u - lambda_l)/s;
        lambda=lambda_u - d_lambda;
      }else{
        s=max(r_l/r - 1.0, 0.1);
        d_lambda=(lambda - lambda_l)/s;
        lambda_new=min(lambda + d_lambda, 
                              0.75*lambda_u + 0.25*lambda);
        lambda_l=lambda;
        r_l=r;
        lambda=lambda_new;
        s=(lambda_u - lambda_l) / (lambda_u-lambda);
      }
    }
    r=phi(x, a, b, z, l, u, lambda, n);
  }
  
  if(inner_iter >= max_iter)
    info("WARNING: DaiFletcher max iterations \n");
  
  
  return inner_iter;

}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
	
	double Gmax=-INF;
	double Gmax2=-INF;
	int Gmax_idx=-1;
	int Gmin_idx=-1;
	double obj_diff_min=INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)	
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmax)
				{
					Gmax=-G[t];
					Gmax_idx=t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmax)
				{
					Gmax=G[t];
					Gmax_idx=t;
				}
		}

	int i=Gmax_idx;
	const Qfloat *Q_i=NULL;
	if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i=Q->get_Q(i,active_size);

	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))
			{
				double grad_diff=Gmax+G[j];
				if (G[j] >= Gmax2)
					Gmax2=G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef=Q_i[i]+QD[j]-2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff=-(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff=-(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min=obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff= Gmax-G[j];
				if (-G[j] >= Gmax2)
					Gmax2=-G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef=Q_i[i]+QD[j]+2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff=-(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff=-(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min=obj_diff;
					}
				}
			}
		}
	}
  // fprintf(stderr, "Gmax=%f Gmax_idx=%d Gmin=%f Gmin_idx=%d \n", Gmax, Gmax_idx, Gmax2, Gmin_idx); 

	if(Gmax+Gmax2 < eps)
		return 1;
  
	out_i=Gmax_idx;
	out_j=Gmin_idx;
	return 0;
}


// return 1 if already optimal, return 0 otherwise
int Solver::qpqp_select_working_set(int &out_i, int &out_j)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
	
	double Gmax=-INF;
	double Gmax2=-INF;
	int Gmax_idx=-1;
	int Gmin_idx=-1;
	double obj_diff_min=INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)	
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmax)
				{
					Gmax=-G[t];
					Gmax_idx=t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmax)
				{
					Gmax=G[t];
					Gmax_idx=t;
				}
		}

	int i=Gmax_idx;
	const Qfloat *Q_i=NULL;
	if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i=Q->get_Q(i,active_size);

	Qfloat *QQ_i = new Qfloat[active_size];				// modified for windows
	Qfloat *QQD = new Qfloat[active_size];				// modified for windows
  
  for(int j=0; j<active_size; j++)
  {
    QQD[j]=0.0;
    QQ_i[j]=0.0;
    for(int n=0; n<num_kernels; n++)
    {
      QQ_i[j]+=(Qfloat) (Qalpha_all[i][n]*Qalpha_all[j][n]);
      QQD[j]+=(Qfloat) (Qalpha_all[j][n]*Qalpha_all[j][n]);
    }
  }

  for(int j=0; j<active_size; j++)
  {
    QQD[j]=(Qfloat) (QD[j] + (QQD[j]/lambda));
    QQ_i[j]=(Qfloat) (Q_i[j] + (QQ_i[j]/lambda));
  }
	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))
			{
				double grad_diff=Gmax+G[j];
				if (G[j] >= Gmax2)
					Gmax2=G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef=QQ_i[i]+QQD[j]-2.0*y[i]*QQ_i[j];
					if (quad_coef > 0)
						obj_diff=-(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff=-(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min=obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff= Gmax-G[j];
				if (-G[j] >= Gmax2)
					Gmax2=-G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef=QQ_i[i]+QQD[j]+2.0*y[i]*QQ_i[j];
					if (quad_coef > 0)
						obj_diff=-(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff=-(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min=obj_diff;
					}
				}
			}
		}
	}
  info("gap=%f ", Gmax+Gmax2); 

	delete [] QQ_i; // modified for windows
	delete [] QQD; // modified for windows
	
	if(Gmax+Gmax2 < eps)
		return 1;
  
	out_i=Gmax_idx;
	out_j=Gmin_idx;
	return 0;
}



// return 1 if already optimal, return 0 otherwise
int Solver::lpqp_select_working_set(int &out_i, int &out_j)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
	double Gmax=-INF;
	double Gmax2=-INF;
	int Gmax_idx=-1;
	int Gmin_idx=-1;
	double obj_diff_min=INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)	
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmax)
				{
					Gmax=-G[t];
					Gmax_idx=t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmax)
				{
					Gmax=G[t];
					Gmax_idx=t;
				}
		}

	int i=Gmax_idx;
  
  double T1=0.0;
  double T11=0.0;
  double gmax=-INF;
  for(int n=0;n<num_kernels;n++)
    if(-grad_d[n] > gmax) gmax=-grad_d[n];
  
  for(int n=0;n<num_kernels;n++)
    if(-grad_d[n]>DTAU)
      T1+=pow((-grad_d[n]/gmax), (double) L_q);
  
  if(T1>0)
    T11=(2-L_q)*lambda*pow(T1,-2.0/L_q)/(gmax*gmax);
  
	
  const Qfloat *Q_i=NULL;
	if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i=Q->get_Q(i,active_size);
  
  Qfloat *QQ_i = new Qfloat[active_size];
	Qfloat *QQD = new Qfloat[active_size];

  double ti = G[i]-p[i];
  
  for(int j=0; j<active_size; j++)
  {
    QQD[j]=0.0;
    QQ_i[j]=0.0;
    if(( (y[j]==+1 && !is_lower_bound(j) && Gmax+G[j]>0) || 
         (y[j]==-1 && !is_upper_bound(j) && Gmax-G[j]>0)) || 
       j==i)
    {
      for(int n=0; n<num_kernels; n++)
      {
        if((-grad_d[n]>DTAU) &&(d[n]>DTAU))
        {
          double tmp=d[n]*Qalpha_all[j][n]/grad_d[n];
          QQD[j]-=(Qfloat) (tmp*Qalpha_all[j][n]);
          QQ_i[j]-=(Qfloat) (tmp*Qalpha_all[i][n]);
        }
      }
      if(T1>DTAU)
      {
        double tj = G[j]-p[j];
        QQD[j]=(Qfloat) (QD[j]+T11*tj*tj+(L_q-1)*QQD[j]);
        QQ_i[j]=(Qfloat)(Q_i[j]+T11*tj*ti+(L_q-1)*QQ_i[j]);
      }
      else
      {
        QQD[j]=(Qfloat) (QD[j]+(L_q-1)*QQD[j]);
        QQ_i[j]=(Qfloat)(Q_i[j]+(L_q-1)*QQ_i[j]);
      }
    }
  }
  
  //assert(fabs(QQ_i[i]-QQD[i])<1e-5);
  double eta_max=0;
  double dir_grad;
  double delta_alpha=0;
	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))
			{
				double grad_diff=Gmax+G[j];
				if (G[j] >= Gmax2)
					Gmax2=G[j];
				if (grad_diff > 0)
				{
					eta_max=geteta(i,j,dir_grad);
					double obj_diff; 
					double quad_coef=QQD[i]+QQD[j]-2.0*y[i]*QQ_i[j];
					if (quad_coef > 0){
						delta_alpha=-dir_grad/quad_coef;
						obj_diff=-(grad_diff*grad_diff)/(2*quad_coef);
					}
					else{
						obj_diff=-(grad_diff*grad_diff)/TAU;
						delta_alpha=-dir_grad/TAU;
					}
					if(delta_alpha>eta_max || delta_alpha<0){
						if(0.5*quad_coef*eta_max*eta_max + dir_grad*eta_max < 0)
							obj_diff=0.5*quad_coef*eta_max*eta_max + dir_grad*eta_max;
							else
							obj_diff=0;							
					}
					if (obj_diff <= obj_diff_min)
					{	
						Gmin_idx=j;
						obj_diff_min=obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff= Gmax-G[j];
				if (-G[j] >= Gmax2)
					Gmax2=-G[j];
				if (grad_diff > 0)
				{
					eta_max=geteta(i,j,dir_grad);
					double obj_diff; 
					double quad_coef=QQD[i]+QQD[j]+2.0*y[i]*QQ_i[j];
					if (quad_coef > 0){
						delta_alpha=-dir_grad/quad_coef;
						obj_diff=-(grad_diff*grad_diff)/(quad_coef*2);
					}
					else{
						delta_alpha=-dir_grad/TAU;
						obj_diff=-(grad_diff*grad_diff)/TAU;
					}
					
					if(delta_alpha>eta_max || delta_alpha<0){
						if(0.5*quad_coef*eta_max*eta_max + dir_grad*eta_max < 0)
							obj_diff=0.5*quad_coef*eta_max*eta_max + dir_grad*eta_max;
							else
							obj_diff=0;							
					}

					if (obj_diff <= obj_diff_min)
					{	
						Gmin_idx=j;
						obj_diff_min=obj_diff;
					}
				}
			}
		}
	}

  info("gap=%f ", Gmax+Gmax2); 

	delete [] QQ_i;
	delete [] QQD;
	
	if(Gmax+Gmax2 < eps)
		return 1;
  
	out_i=Gmax_idx;
	out_j=Gmin_idx;
	

	
	return 0;
}



// return 1 if already optimal, return 0 otherwise
int Solver::ent_select_working_set(int &out_i, int &out_j)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
	
	double Gmax=-INF;
	double Gmax2=-INF;
	int Gmax_idx=-1;
	int Gmin_idx=-1;
	double obj_diff_min=INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)	
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmax)
				{
					Gmax=-G[t];
					Gmax_idx=t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmax)
				{
					Gmax=G[t];
					Gmax_idx=t;
				}
		}

	int i=Gmax_idx;
	const Qfloat *Q_i=NULL;
	if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i=Q->get_Q(i,active_size);

    Qfloat *QQ_i = new Qfloat[active_size];				// modified for windows
	Qfloat *QQD = new Qfloat[active_size];				// modified for windows
	Qfloat *Qalpha = new Qfloat[active_size];			// modified for windows
  
  for(int j=0; j<active_size; j++)
  {
    QQD[j]=0.0;
    QQ_i[j]=0.0;
    Qalpha[j]=0.0;
    for(int n=0; n<num_kernels; n++)
      if(d[n] > DTAU)
      {
        QQ_i[j]+=(Qfloat) (d[n]*Qalpha_all[i][n]*Qalpha_all[j][n]);
        QQD[j]+=(Qfloat) (d[n]*Qalpha_all[j][n]*Qalpha_all[j][n]);
        Qalpha[j]+=(Qfloat) (d[n]*Qalpha_all[j][n]); 
      }
  }
  
  for(int j=0; j<active_size; j++)
  {
    QQD[j]= (Qfloat) (QD[j] + ((QQD[j] - Qalpha[j]*Qalpha[j])/lambda));
    QQ_i[j]=(Qfloat) (Q_i[j] + ((QQ_i[j] - Qalpha[i]*Qalpha[j])/lambda));
  }
  
	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))
			{
				double grad_diff=Gmax+G[j];
				if (G[j] >= Gmax2)
					Gmax2=G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef=QQ_i[i]+QQD[j]-2.0*y[i]*QQ_i[j];
					if (quad_coef > 0)
						obj_diff=-(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff=-(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min=obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff= Gmax-G[j];
				if (-G[j] >= Gmax2)
					Gmax2=-G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef=QQ_i[i]+QQD[j]+2.0*y[i]*QQ_i[j];
					if (quad_coef > 0)
						obj_diff=-(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff=-(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min=obj_diff;
					}
				}
			}
		}
	}
  // fprintf(stderr, "Gmax=%f Gmax_idx=%d Gmin=%f Gmin_idx=%d \n", Gmax, Gmax_idx, Gmax2, Gmin_idx); 
  
  	delete [] QQ_i;										// modified for windows
	delete [] QQD;										// modified for windows
	delete [] Qalpha;									// modified for windows
	info("gap=%f ",Gmax+Gmax2);
	if(Gmax+Gmax2 < eps)
		return 1;
  
	out_i=Gmax_idx;
	out_j=Gmin_idx;

	return 0;
}

double Solver::geteta(int i, int j, double &dir_grad){
	double eta_max=0;
	double C_i=get_C(i);
    double C_j=get_C(j);
if(y[i]==y[j])
    {
      if(G[i]-G[j]<0)
      {
		  dir_grad=G[i]-G[j];
        eta_max=min(C_i-alpha[i], alpha[j]);
      }
      else
      {
		  dir_grad=G[j]-G[i];
        eta_max=min(alpha[i], C_j-alpha[j]);
      }
    }
    else
    {
      if(G[i]+G[j]<0)
      {
		  dir_grad=G[i]+G[j];
        eta_max=min(C_i-alpha[i], C_j-alpha[j]);
      }
      else
      {
		  dir_grad=-G[i]-G[j];
        eta_max=min(alpha[i], alpha[j]);
      }
    }
    return eta_max;
}

bool Solver::be_shrunk(int i, double Gmax1, double Gmax2)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else
			return(-G[i] > Gmax2);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax1);
	}
	else
		return(false);
}

void Solver::do_shrinking()
{
	int i;
	double Gmax1=-INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	double Gmax2=-INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for(i=0;i<active_size;i++)
	{
		if(y[i]==+1)	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax1)
					Gmax1=-G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax2)
					Gmax2=G[i];
			}
		}
		else	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax2)
					Gmax2=-G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax1)
					Gmax1=G[i];
			}
		}
	}

	if(unshrink == false && Gmax1 + Gmax2 <= eps*10) 
	{
		unshrink=true;
		reconstruct_gradient();
		active_size=l;
		// info("*");
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

double Solver::calculate_rho()
{
	double r;
	int nr_free=0;
	double ub=INF, lb=-INF, sum_free=0;
	for(int i=0;i<active_size;i++)
	{
		double yG=y[i]*G[i];

		if(is_upper_bound(i))
		{
			if(y[i]==-1)
				ub=min(ub,yG);
			else
				lb=max(lb,yG);
		}
		else if(is_lower_bound(i))
		{
			if(y[i]==+1)
				ub=min(ub,yG);
			else
				lb=max(lb,yG);
		}
		else
		{
			++nr_free;
			sum_free+=yG;
		}
	}

	if(nr_free>0)
		r=sum_free/nr_free;
	else
		r=(ub+lb)/2;

	return r;
}


//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha=constant
//
class Solver_NU : public Solver
{
public:
	Solver_NU() {}
	void Solve(int l, float L_p, const QMatrix& Q, const double *p, const schar *y,
		   double *alpha, double Cp, double Cn, double eps,
             SolutionInfo* si, int shrinking, int solver_type, int d_regularizer, 
             int d_proj, double lambda_, double obj_threshold_, double diff_threshold_)
	{
		this->si=si;
		Solver::Solve(l,L_p,Q,p,y,alpha,Cp,Cn,eps,si,shrinking,solver_type, d_regularizer, d_proj, lambda_,obj_threshold_,diff_threshold_);
	}
private:
	SolutionInfo *si;
	int select_working_set(int &i, int &j);
	double calculate_rho();
	bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
	void do_shrinking();
};

// return 1 if already optimal, return 0 otherwise
int Solver_NU::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that y_i=y_j and
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmaxp=-INF;
	double Gmaxp2=-INF;
	int Gmaxp_idx=-1;

	double Gmaxn=-INF;
	double Gmaxn2=-INF;
	int Gmaxn_idx=-1;

	int Gmin_idx=-1;
	double obj_diff_min=INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmaxp)
				{
					Gmaxp=-G[t];
					Gmaxp_idx=t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmaxn)
				{
					Gmaxn=G[t];
					Gmaxn_idx=t;
				}
		}

	int ip=Gmaxp_idx;
	int in=Gmaxn_idx;
	const Qfloat *Q_ip=NULL;
	const Qfloat *Q_in=NULL;
	if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
		Q_ip=Q->get_Q(ip,active_size);
	if(in != -1)
		Q_in=Q->get_Q(in,active_size);

	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))	
			{
				double grad_diff=Gmaxp+G[j];
				if (G[j] >= Gmaxp2)
					Gmaxp2=G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef=Q_ip[ip]+QD[j]-2*Q_ip[j];
					if (quad_coef > 0)
						obj_diff=-(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff=-(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min=obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff=Gmaxn-G[j];
				if (-G[j] >= Gmaxn2)
					Gmaxn2=-G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef=Q_in[in]+QD[j]-2*Q_in[j];
					if (quad_coef > 0)
						obj_diff=-(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff=-(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min=obj_diff;
					}
				}
			}
		}
	}

	if(max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2) < eps)
		return 1;

	if (y[Gmin_idx] == +1)
		out_i=Gmaxp_idx;
	else
		out_i=Gmaxn_idx;
	out_j=Gmin_idx;

	return 0;
}

bool Solver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else	
			return(-G[i] > Gmax4);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax3);
	}
	else
		return(false);
}

void Solver_NU::do_shrinking()
{
	double Gmax1=-INF;	// max { -y_i * grad(f)_i | y_i=+1, i in I_up(\alpha) }
	double Gmax2=-INF;	// max { y_i * grad(f)_i | y_i=+1, i in I_low(\alpha) }
	double Gmax3=-INF;	// max { -y_i * grad(f)_i | y_i=-1, i in I_up(\alpha) }
	double Gmax4=-INF;	// max { y_i * grad(f)_i | y_i=-1, i in I_low(\alpha) }

	// find maximal violating pair first
	int i;
	for(i=0;i<active_size;i++)
	{
		if(!is_upper_bound(i))
		{
			if(y[i]==+1)
			{
				if(-G[i] > Gmax1) Gmax1=-G[i];
			}
			else	if(-G[i] > Gmax4) Gmax4=-G[i];
		}
		if(!is_lower_bound(i))
		{
			if(y[i]==+1)
			{	
				if(G[i] > Gmax2) Gmax2=G[i];
			}
			else	if(G[i] > Gmax3) Gmax3=G[i];
		}
	}

	if(unshrink == false && max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10) 
	{
		unshrink=true;
		reconstruct_gradient();
		active_size=l;
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

double Solver_NU::calculate_rho()
{
	int nr_free1=0,nr_free2=0;
	double ub1=INF, ub2=INF;
	double lb1=-INF, lb2=-INF;
	double sum_free1=0, sum_free2=0;

	for(int i=0;i<active_size;i++)
	{
		if(y[i]==+1)
		{
			if(is_upper_bound(i))
				lb1=max(lb1,G[i]);
			else if(is_lower_bound(i))
				ub1=min(ub1,G[i]);
			else
			{
				++nr_free1;
				sum_free1+=G[i];
			}
		}
		else
		{
			if(is_upper_bound(i))
				lb2=max(lb2,G[i]);
			else if(is_lower_bound(i))
				ub2=min(ub2,G[i]);
			else
			{
				++nr_free2;
				sum_free2+=G[i];
			}
		}
	}

	double r1,r2;
	if(nr_free1 > 0)
		r1=sum_free1/nr_free1;
	else
		r1=(ub1+lb1)/2;
	
	if(nr_free2 > 0)
		r2=sum_free2/nr_free2;
	else
		r2=(ub2+lb2)/2;
	
	si->r=(r1+r2)/2;
	return (r1-r2)/2;
}

//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{ 
public:
	SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
	:Kernel(prob.l, prob.x, param)
	{
		clone(y,y_,l);
		master_cache=new Cache(l,(long int)(param.cache_size*(1<<20)));
		cache=new Cache*[num_kernels];
    for(int n=0;n<num_kernels;n++)
			cache[n]=new Cache(l,(long int)(param.cache_size*(1<<20)));
		QD=new Qfloat[l];
		QD_all=new Qfloat*[l];
		for(int i=0;i<l;i++)
    {
      QD[i]=0.0;
      QD_all[i]=new Qfloat[num_kernels];
      for(int n=0;n<num_kernels;n++)
      {
        Qfloat k_value=(Qfloat) kernel_all(n,i,i); 
        QD_all[i][n]=k_value;
        if(d[n] > DTAU)
          QD[i]+=(Qfloat)(k_value*d[n]);
      }
    }
	}

	Qfloat *get_Qn(int n, int i, int len) const
	{
		Qfloat *data;
		int start, j;
    
    if((start=cache[n]->get_data(i,&data,len)) < len)
    {
      for(j=start;j<len;j++)
        data[j]=(Qfloat)(y[i]*y[j]*(this->kernel_all)(n,i,j));
    }
    
		return data;
	}
	
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *master_data,*data;

		master_cache->get_data(i,&master_data,len);
		for(int j=0;j<len;j++)
			master_data[j]=0.0;

		for(int n=0;n<num_kernels;n++)
		{
      data=get_Qn(n, i, len);
			for(int j=0;j<len;j++)
        if(d[n] > DTAU)
          master_data[j]+= (Qfloat)(d[n]*data[j]);
		}
    
		return master_data;
	}


	Qfloat *get_QD() const
	{
		// svnvish: BUGBUG
    // optimize by calculating on demand
    for(int i=0;i<l;i++)
    {
      QD[i]=0.0;
      for(int n=0;n<num_kernels;n++)
        if(d[n] > DTAU)
          QD[i]+=(Qfloat)(QD_all[i][n]*d[n]);
    }
		return QD;
	}

	void swap_index(int i, int j) const
	{
		for(int n=0;n<num_kernels;n++)
			cache[n]->swap_index(i,j);
    master_cache->swap_index(i, j);
		Kernel::swap_index(i,j);
		swap(y[i],y[j]);
		swap(QD[i],QD[j]);
		swap(QD_all[i],QD_all[j]);
	}

	~SVC_Q()
	{
		delete[] y;
	}
private:
	schar *y;
};

class ONE_CLASS_Q: public Kernel
{
public:
	ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		master_cache=new Cache(l,(long int)(param.cache_size*(1<<20)));
    cache=new Cache*[num_kernels];
		for(int n=0;n<num_kernels;n++)
		  cache[n]=new Cache(l,(long int)(param.cache_size*(1<<20)));
		QD=new Qfloat[l];
		QD_all=new Qfloat*[l];
		for(int i=0;i<l;i++)
    {
      QD[i]=0.0;
      QD_all[i]=new Qfloat[num_kernels];
      for(int n=0;n<num_kernels;n++)
      {
        Qfloat k_value=(Qfloat) kernel_all(n,i,i); 
        QD_all[i][n]=k_value;
        if(d[n] > DTAU)
          QD[i]+=(Qfloat)(k_value*d[n]);
      }
    }
	}
	
	Qfloat *get_Qn(int n, int i, int len) const
	{
		Qfloat *data;
		int start, j;
    
    if((start=cache[n]->get_data(i,&data,len)) < len)
			{
				for(j=start;j<len;j++)
          data[j]=(Qfloat)((this->kernel_all)(n,i,j));
			}
    
		return data;
	}

	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *master_data,*data;
    
    master_cache->get_data(i,&master_data,len);
    for(int j=0;j<len;j++)
      master_data[j]=0.0;

    for(int n=0;n<num_kernels;n++)
    {
      data=get_Qn(n, i, len);
      for(int j=0;j<len;j++)
        if(d[n] > DTAU)
          master_data[j]+= (Qfloat)(d[n]*data[j]);
    }
    return master_data;
	}

	Qfloat *get_QD() const
	{
    // svnvish: BUGBUG
    // optimize by calculating on demand
		for(int i=0;i<l;i++)
    {
      QD[i]=0.0;
      for(int n=0;n<num_kernels;n++)
        if(d[n] > DTAU)
          QD[i]+=(Qfloat)(QD_all[i][n]*d[n]);
    }
    
		return QD;
	}

	void swap_index(int i, int j) const
	{
		for(int n=0;n<num_kernels;n++)
			cache[n]->swap_index(i,j);
    master_cache->swap_index(i, j);
		Kernel::swap_index(i,j);
		swap(QD[i],QD[j]);
    swap(QD_all[i],QD_all[j]); 
  }

	~ONE_CLASS_Q()
	{}
private:
};

class SVR_Q: public Kernel
{ 
public:
	SVR_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		master_cache=new Cache(l,(long int)(param.cache_size*(1<<20)));
    cache=new Cache*[num_kernels];
		for(int n=0;n<num_kernels;n++)
			cache[n]=new Cache(l,(long int)(param.cache_size*(1<<20)));
		QD=new Qfloat[2*l];
		sign=new schar[2*l];
		index=new int[2*l];
		for(int k=0;k<l;k++)
		{
			sign[k]=1;
			sign[k+l]=-1;
			index[k]=k;
			index[k+l]=k;
		}

		QD=new Qfloat[2*l];
		QD_all=new Qfloat*[l];
		for(int i=0;i<l;i++)
    {
      QD[i]=0.0;
      QD_all[i]=new Qfloat[num_kernels];
      for(int n=0;n<num_kernels;n++)
      {
        Qfloat k_value=(Qfloat) kernel_all(n,i,i); 
        QD_all[i][n]=k_value;
        if(d[n] > DTAU)
          QD[i]+=(Qfloat)(k_value*d[n]);
      }
      QD[i+l]=QD[i];
    }
		buffer[0]=new Qfloat[2*l];
		buffer[1]=new Qfloat[2*l];
		next_buffer=0;
	}

	void swap_index(int i, int j) const
	{
		swap(sign[i],sign[j]);
		swap(index[i],index[j]);
		swap(QD[i],QD[j]);
    swap(QD_all[i],QD_all[j]);
	}
	
	Qfloat *get_Qn(int n, int i, int len) const
	{
		Qfloat *data;
		int start, j, real_i=index[i];
    
    if((start=cache[n]->get_data(real_i,&data,l)) < l)
			{
				for(j=start;j<l;j++)
          data[j]=(Qfloat)((this->kernel_all)(n,real_i,j));
			}
    
		// reorder and copy
		Qfloat *buf=buffer[next_buffer];
		next_buffer=1 - next_buffer;
		schar si=sign[i];
		for(j=0;j<len;j++)
			buf[j]=(Qfloat) si * (Qfloat) sign[j] * data[index[j]];
		return buf;
  }

	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *master_data,*data;
		int real_i=index[i];
		int start;

    master_cache->get_data(real_i,&master_data,l);
    for(int j=0;j<l;j++)
      master_data[j]=0.0;

    for(int n=0;n<num_kernels;n++)
    {
	if(d[n] > DTAU){
      if((start = cache[n]->get_data(real_i,&data,l)) < l)
      {
        for(int j=start;j<l;j++)
          data[j]=(Qfloat)((this->kernel_all)(n,real_i,j));
      }
      for(int j=0;j<l;j++)
          master_data[j]+=(Qfloat)(d[n]*data[j]);
	  }
    }

		// reorder and copy
		Qfloat *buf=buffer[next_buffer];
		next_buffer=1 - next_buffer;
		schar si=sign[i];
		for(int j=0;j<len;j++)
			buf[j]=(Qfloat) si * (Qfloat) sign[j] * master_data[index[j]];
		return buf;
	}

	Qfloat *get_QD() const
	{
        
    // svnvish: BUGBUG
    // optimize by calculating on demand
		for(int i=0;i<l;i++)
    {
      QD[i]=0.0;
      for(int n=0;n<num_kernels;n++)
        if(d[n] > DTAU)
          QD[i]+=(Qfloat)(QD_all[i][n]*d[n]);
      QD[i+l]=QD[i];
    }
    
		return QD;
	}

	~SVR_Q()
	{
		delete[] sign;
		delete[] index;
		delete[] buffer[0];
		delete[] buffer[1];
	}
private:
	schar *sign;
	int *index;
	mutable int next_buffer;
	Qfloat *buffer[2];
};

//
// construct and solve various formulations
//
static void solve_c_svc(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
	lookuptable();
	int l=prob->l;
	double *minus_ones=new double[l];
	schar *y=new schar[l];

	int i;

	for(i=0;i<l;i++)
	{
		alpha[i]=0;
		minus_ones[i]=-1;
		if(prob->y[i] > 0) y[i]=+1; else y[i]=-1;
	}

	Solver s;
				
	SVC_Q *tmp=new SVC_Q(*prob,*param,y);
	double *scale_factor=tmp->get_scale_factor();
	s.Solve(l, param->L_p, *tmp, minus_ones, y,
          alpha, Cp, Cn, param->eps, si, param->shrinking, 
          param->solver_type, param->d_regularizer,
          param->d_proj, param->lambda, param->obj_threshold, param->diff_threshold);
	
	double *d=tmp->get_d();
	for(i=0;i<tmp->get_num_kernels();i++){
		param->kernels[i].coef=d[i];
		param->kernels[i].scale_factor=scale_factor[i];
	}

	double sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha+=alpha[i];

	if (Cp==Cn)
		info("nu=%f\n", sum_alpha/(Cp*prob->l));

	for(i=0;i<l;i++)
		alpha[i] *= y[i];

	delete tmp;
	delete[] minus_ones;
	delete[] y;
}

static void solve_nu_svc(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int i;
	int l=prob->l;
	double nu=param->nu;

	schar *y=new schar[l];

	for(i=0;i<l;i++)
		if(prob->y[i]>0)
			y[i]=+1;
		else
			y[i]=-1;

	double sum_pos=nu*l/2;
	double sum_neg=nu*l/2;

	for(i=0;i<l;i++)
		if(y[i] == +1)
		{
			alpha[i]=min(1.0,sum_pos);
			sum_pos -= alpha[i];
		}
		else
		{
			alpha[i]=min(1.0,sum_neg);
			sum_neg -= alpha[i];
		}

	double *zeros=new double[l];

	for(i=0;i<l;i++)
		zeros[i]=0;

	Solver_NU s;
	SVC_Q *tmp=new SVC_Q(*prob,*param,y);
	s.Solve(l, param->L_p, *tmp, zeros, y,
          alpha, 1.0, 1.0, param->eps, si,  param->shrinking, 
          param->solver_type, param->d_regularizer,
          param->d_proj, param->lambda, param->obj_threshold, param->diff_threshold);
	
	double *d=tmp->get_d();
	for(i=0;i<tmp->get_num_kernels();i++)
		param->kernels[i].coef=d[i];
	double r=si->r;

	info("C=%f\n",1/r);

	for(i=0;i<l;i++)
		alpha[i] *= y[i]/r;

	si->rho /= r;
	si->obj /= (r*r);
	si->upper_bound_p=1/r;
	si->upper_bound_n=1/r;

	delete tmp;
	delete[] y;
	delete[] zeros;
}

static void solve_one_class(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l=prob->l;
	double *zeros=new double[l];
	schar *ones=new schar[l];
	int i;

	int n=(int)(param->nu*prob->l);	// # of alpha's at upper bound

	for(i=0;i<n;i++)
		alpha[i]=1;
	if(n<prob->l)
		alpha[n]=param->nu * prob->l - n;
	for(i=n+1;i<l;i++)
		alpha[i]=0;

	for(i=0;i<l;i++)
	{
		zeros[i]=0;
		ones[i]=1;
	}

	Solver s;
	ONE_CLASS_Q *tmp=new ONE_CLASS_Q(*prob,*param);
	s.Solve(l, param->L_p, *tmp, zeros, ones,
          alpha, 1.0, 1.0, param->eps, si, param->shrinking, param->solver_type, 
          param->d_regularizer, param->d_proj, 
          param->lambda, param->obj_threshold, param->diff_threshold);

	double *d=tmp->get_d();
	for(i=0;i<tmp->get_num_kernels();i++)
		param->kernels[i].coef=d[i];

	delete tmp;
	delete[] zeros;
	delete[] ones;
}

static void solve_epsilon_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l=prob->l;
	double *alpha2=new double[2*l];
	double *linear_term=new double[2*l];
	schar *y=new schar[2*l];
	int i;

	for(i=0;i<l;i++)
	{
		alpha2[i]=0;
		linear_term[i]=param->p - prob->y[i];
		y[i]=1;

		alpha2[i+l]=0;
		linear_term[i+l]=param->p + prob->y[i];
		y[i+l]=-1;
	}

	Solver s;
	SVR_Q *tmp=new SVR_Q(*prob,*param);
	double *scale_factor=tmp->get_scale_factor();
	s.Solve(2*l, param->L_p, *tmp, linear_term, y,
          alpha2, param->C, param->C, param->eps, si, param->shrinking, param->solver_type, 
          param->d_regularizer, param->d_proj, 
          param->lambda, param->obj_threshold, param->diff_threshold);
  
	double *d=tmp->get_d();
	for(i=0;i<tmp->get_num_kernels();i++){
		param->kernels[i].coef=d[i];
		param->kernels[i].scale_factor=scale_factor[i];
	}

	double sum_alpha=0;
	for(i=0;i<l;i++)
	{
		alpha[i]=alpha2[i] - alpha2[i+l];
		sum_alpha+=fabs(alpha[i]);
	}
	info("nu=%f\n",sum_alpha/(param->C*l));

	delete tmp;
	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

static void solve_nu_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l=prob->l;
	double C=param->C;
	double *alpha2=new double[2*l];
	double *linear_term=new double[2*l];
	schar *y=new schar[2*l];
	int i;

	double sum=C * param->nu * l / 2;
	for(i=0;i<l;i++)
	{
		alpha2[i]=alpha2[i+l]=min(sum,C);
		sum -= alpha2[i];

		linear_term[i]=- prob->y[i];
		y[i]=1;

		linear_term[i+l]=prob->y[i];
		y[i+l]=-1;
	}

	Solver_NU s;
	SVR_Q *tmp=new SVR_Q(*prob,*param);
	s.Solve(2*l, param->L_p, *tmp, linear_term, y,
          alpha2, C, C, param->eps, si, param->shrinking, param->solver_type, 
          param->d_regularizer, param->d_proj, 
          param->lambda, param->obj_threshold, param->diff_threshold);
	
	double *d=tmp->get_d();
	for(i=0;i<tmp->get_num_kernels();i++)
		param->kernels[i].coef=d[i];

	info("epsilon=%f\n",-si->r);

	for(i=0;i<l;i++)
		alpha[i]=alpha2[i] - alpha2[i+l];

	delete tmp;
	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

//
// decision_function
//
struct decision_function
{
	double *alpha;
	double rho;	
};

static decision_function svm_train_one(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn)
{
	double *alpha=Malloc(double,prob->l);
	Solver::SolutionInfo si;
	si.rho=0.0;
  si.obj=0.0;
  switch(param->svm_type)
	{
		case C_SVC:
			solve_c_svc(prob,param,alpha,&si,Cp,Cn);
			break;
		case NU_SVC:
			solve_nu_svc(prob,param,alpha,&si);
			break;
		case ONE_CLASS:
			solve_one_class(prob,param,alpha,&si);
			break;
		case EPSILON_SVR:
			solve_epsilon_svr(prob,param,alpha,&si);
			break;
		case NU_SVR:
			solve_nu_svr(prob,param,alpha,&si);
			break;
	}

	info("obj=%f, rho=%f\n",si.obj,si.rho);

	// output SVs

	int nSV=0;
	int nBSV=0;
	for(int i=0;i<prob->l;i++)
	{
		if(fabs(alpha[i]) > 0)
		{
			++nSV;
			if(prob->y[i] > 0)
			{
				if(fabs(alpha[i]) >= si.upper_bound_p)
					++nBSV;
			}
			else
			{
				if(fabs(alpha[i]) >= si.upper_bound_n)
					++nBSV;
			}
		}
	}

	info("nSV=%d, nBSV=%d\n",nSV,nBSV);

	decision_function f;
	f.alpha=alpha;
	f.rho=si.rho;
	return f;
}

//
// svm_model
//
struct svm_model
{
	svm_parameter *param;	// array of parameter structures
	int nr_class;		// number of classes,=2 in regression/one class svm
	int l;			// total #SV
	svm_node **SV;		// SVs (SV[l])
	double **sv_coef;	// coefficients for SVs in decision functions (sv_coef[k-1][l])
	double *rho;		// constants in decision functions (rho[k*(k-1)/2])
	double *probA;		// pariwise probability information
	double *probB;

	// for classification only

	int *label;		// label of each class (label[k])
	int *nSV;		// number of SVs for each class (nSV[k])
				// nSV[0] + nSV[1] + ... + nSV[k-1]=l
	// XXX
	int free_sv;		// 1 if svm_model is created by svm_load_model
				// 0 if svm_model is created by svm_train
};

// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
static void sigmoid_train(int l, const double *dec_values, const double *labels, 
	double& A, double& B)
{
	double prior1=0, prior0=0;
	int i;

	for (i=0;i<l;i++)
		if (labels[i] > 0) prior1+=1;
		else prior0+=1;
	
	int max_iter=100;	// Maximal number of iterations
	double min_step=1e-10;	// Minimal step taken in line search
	double sigma=1e-12;	// For numerically strict PD of Hessian
	double eps=1e-5;
	double hiTarget=(prior1+1.0)/(prior1+2.0);
	double loTarget=1/(prior0+2.0);
	double *t=Malloc(double,l);
	double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
	double newA,newB,newf,d1,d2;
	int iter; 
	
	// Initial Point and Initial Fun Value
	A=0.0; B=log((prior0+1.0)/(prior1+1.0));
	double fval=0.0;

	for (i=0;i<l;i++)
	{
		if (labels[i]>0) t[i]=hiTarget;
		else t[i]=loTarget;
		fApB=dec_values[i]*A+B;
		if (fApB>=0)
			fval+=t[i]*fApB + log(1+exp(-fApB));
		else
			fval+=(t[i] - 1)*fApB +log(1+exp(fApB));
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// Update Gradient and Hessian (use H'=H + sigma I)
		h11=sigma; // numerically ensures strict PD
		h22=sigma;
		h21=0.0;g1=0.0;g2=0.0;
		for (i=0;i<l;i++)
		{
			fApB=dec_values[i]*A+B;
			if (fApB >= 0)
			{
				p=exp(-fApB)/(1.0+exp(-fApB));
				q=1.0/(1.0+exp(-fApB));
			}
			else
			{
				p=1.0/(1.0+exp(fApB));
				q=exp(fApB)/(1.0+exp(fApB));
			}
			d2=p*q;
			h11+=dec_values[i]*dec_values[i]*d2;
			h22+=d2;
			h21+=dec_values[i]*d2;
			d1=t[i]-p;
			g1+=dec_values[i]*d1;
			g2+=d1;
		}

		// Stopping Criteria
		if (fabs(g1)<eps && fabs(g2)<eps)
			break;

		// Finding Newton direction: -inv(H') * g
		det=h11*h22-h21*h21;
		dA=-(h22*g1 - h21 * g2) / det;
		dB=-(-h21*g1+ h11 * g2) / det;
		gd=g1*dA+g2*dB;


		stepsize=1;		// Line Search
		while (stepsize >= min_step)
		{
			newA=A + stepsize * dA;
			newB=B + stepsize * dB;

			// New function value
			newf=0.0;
			for (i=0;i<l;i++)
			{
				fApB=dec_values[i]*newA+newB;
				if (fApB >= 0)
					newf+=t[i]*fApB + log(1+exp(-fApB));
				else
					newf+=(t[i] - 1)*fApB +log(1+exp(fApB));
			}
			// Check sufficient decrease
			if (newf<fval+0.0001*stepsize*gd)
			{
				A=newA;B=newB;fval=newf;
				break;
			}
			else
				stepsize=stepsize / 2.0;
		}

		if (stepsize < min_step)
		{
			info("Line search fails in two-class probability estimates\n");
			break;
		}
	}

	if (iter>=max_iter)
		info("Reaching maximal iterations in two-class probability estimates\n");
	free(t);
}

static double sigmoid_predict(double decision_value, double A, double B)
{
	double fApB=decision_value*A+B;
	if (fApB >= 0)
		return exp(-fApB)/(1.0+exp(-fApB));
	else
		return 1.0/(1+exp(fApB)) ;
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
static void multiclass_probability(int k, double **r, double *p)
{
	int t,j;
	int iter=0, max_iter=max(100,k);
	double **Q=Malloc(double *,k);
	double *Qp=Malloc(double,k);
	double pQp, eps=0.005/k;
	
	for (t=0;t<k;t++)
	{
		p[t]=1.0/k;  // Valid if k=1
		Q[t]=Malloc(double,k);
		Q[t][t]=0;
		for (j=0;j<t;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=Q[j][t];
		}
		for (j=t+1;j<k;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=-r[j][t]*r[t][j];
		}
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp=0;
		for (t=0;t<k;t++)
		{
			Qp[t]=0;
			for (j=0;j<k;j++)
				Qp[t]+=Q[t][j]*p[j];
			pQp+=p[t]*Qp[t];
		}
		double max_error=0;
		for (t=0;t<k;t++)
		{
			double error=fabs(Qp[t]-pQp);
			if (error>max_error)
				max_error=error;
		}
		if (max_error<eps) break;
		
		for (t=0;t<k;t++)
		{
			double diff=(-Qp[t]+pQp)/Q[t][t];
			p[t]+=diff;
			pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
			for (j=0;j<k;j++)
			{
				Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
				p[j]/=(1+diff);
			}
		}
	}
	if (iter>=max_iter)
		info("Exceeds max_iter in multiclass_prob\n");
	for(t=0;t<k;t++) free(Q[t]);
	free(Q);
	free(Qp);
}

// Cross-validation decision values for probability estimates
static void svm_binary_svc_probability(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn, double& probA, double& probB)
{
	int i;
	int nr_fold=5;
	int *perm=Malloc(int,prob->l);
	double *dec_values=Malloc(double,prob->l);

	// random shuffle
	for(i=0;i<prob->l;i++) perm[i]=i;
	for(i=0;i<prob->l;i++)
	{
		int j=i+rand()%(prob->l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<nr_fold;i++)
	{
		int begin=i*prob->l/nr_fold;
		int end=(i+1)*prob->l/nr_fold;
		int j,k;
		struct svm_problem subprob;

		subprob.l=prob->l-(end-begin);
		subprob.x=Malloc(struct svm_node*,subprob.l);
		subprob.y=Malloc(double,subprob.l);
			
		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k]=prob->x[perm[j]];
			subprob.y[k]=prob->y[perm[j]];
			++k;
		}
		for(j=end;j<prob->l;j++)
		{
			subprob.x[k]=prob->x[perm[j]];
			subprob.y[k]=prob->y[perm[j]];
			++k;
		}
		int p_count=0,n_count=0;
		for(j=0;j<k;j++)
			if(subprob.y[j]>0)
				p_count++;
			else
				n_count++;

		if(p_count==0 && n_count==0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]]=0;
		else if(p_count > 0 && n_count == 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]]=1;
		else if(p_count == 0 && n_count > 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]]=-1;
		else
		{
			svm_parameter *subparam=new svm_parameter;
			copy_param(subparam,param);
			free(subparam->weight_label);
			free(subparam->weight);
			subparam->probability=0;
			subparam->C=1.0;
			subparam->nr_weight=2;
			subparam->weight_label=Malloc(int,2);
			subparam->weight=Malloc(double,2);
			subparam->weight_label[0]=+1;
			subparam->weight_label[1]=-1;
			subparam->weight[0]=Cp;
			subparam->weight[1]=Cn;
			struct svm_model *submodel=svm_train(&subprob,subparam);
			for(j=begin;j<end;j++)
			{
				svm_predict_values(submodel,prob->x[perm[j]],&(dec_values[perm[j]])); 
				// ensure +1 -1 order; reason not using CV subroutine
				dec_values[perm[j]] *= submodel->label[0];
			}		
			svm_destroy_model(submodel);
			svm_destroy_param(subparam);
			delete subparam;
		}
		free(subprob.x);
		free(subprob.y);
	}		
	sigmoid_train(prob->l,dec_values,prob->y,probA,probB);
	free(dec_values);
	free(perm);
}

// Return parameter of a Laplace distribution 
static double svm_svr_probability(
	const svm_problem *prob, const svm_parameter *param)
{
	int i;
	int nr_fold=5;
	double *ymv=Malloc(double,prob->l);
	double mae=0;

	svm_parameter *newparam=new svm_parameter;
	copy_param(newparam,param);
	newparam->probability=0;
	svm_cross_validation(prob,newparam,nr_fold,ymv);
	for(i=0;i<prob->l;i++)
	{
		ymv[i]=prob->y[i]-ymv[i];
		mae+=fabs(ymv[i]);
	}		
	mae /= prob->l;
	double std=sqrt(2*mae*mae);
	int count=0;
	mae=0;
	for(i=0;i<prob->l;i++)
		if (fabs(ymv[i]) > 5*std) 
			count=count+1;
		else 
			mae+=fabs(ymv[i]);
	mae /= (prob->l-count);
	info("Prob. model for test data: target value=predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae);
	svm_destroy_param(newparam);
	delete newparam;
	free(ymv);
	return mae;
}


// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l=prob->l;
	int max_nr_class=16;
	int nr_class=0;
	int *label=Malloc(int,max_nr_class);
	int *count=Malloc(int,max_nr_class);
	int *data_label=Malloc(int,l);	
	int i;

	for(i=0;i<l;i++)
	{
		int this_label=(int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i]=j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label=(int *)realloc(label,max_nr_class*sizeof(int));
				count=(int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class]=this_label;
			count[nr_class]=1;
			++nr_class;
		}
	}

	int *start=Malloc(int,nr_class);
	start[0]=0;
	for(i=1;i<nr_class;i++)
		start[i]=start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]]=i;
		++start[data_label[i]];
	}
	start[0]=0;
	for(i=1;i<nr_class;i++)
		start[i]=start[i-1]+count[i-1];

	*nr_class_ret=nr_class;
	*label_ret=label;
	*start_ret=start;
	*count_ret=count;
	free(data_label);
}

//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
{
	svm_model *model=Malloc(svm_model,1);

	model->free_sv=0;	// XXX
  
	if(param->svm_type == ONE_CLASS ||
	   param->svm_type == EPSILON_SVR ||
	   param->svm_type == NU_SVR)
	{
		// regression or one-class-svm
		model->nr_class=2;
		model->label=NULL;
		model->nSV=NULL;
		model->probA=NULL; model->probB=NULL;
		model->sv_coef=Malloc(double *,1);

		if(param->probability && 
		   (param->svm_type == EPSILON_SVR ||
		    param->svm_type == NU_SVR))
		{
			model->probA=Malloc(double,1);
			model->probA[0]=svm_svr_probability(prob,param);
		}

		decision_function f=svm_train_one(prob,param,0,0);
		model->param=new svm_parameter[1];
		copy_param(model->param,param);

		model->rho=Malloc(double,1);
		model->rho[0]=f.rho;

		int nSV=0;
		int i;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0) ++nSV;
		model->l=nSV;
		model->SV=Malloc(svm_node *,nSV);
		model->sv_coef[0]=Malloc(double,nSV);
		int j=0;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0)
			{
				model->SV[j]=prob->x[i];
				model->sv_coef[0][j]=f.alpha[i];
				++j;
			}		

		free(f.alpha);
	}
	else
	{
		// classification
		int l=prob->l;
		int nr_class;
		int *label=NULL;
		int *start=NULL;
		int *count=NULL;
		int *perm=Malloc(int,l);

		// group training data of the same class
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);		
		svm_node **x=Malloc(svm_node *,l);
		int i;
		for(i=0;i<l;i++)
			x[i]=prob->x[perm[i]];

		// calculate weighted C

		double *weighted_C=Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i]=param->C;
		for(i=0;i<param->nr_weight;i++)
		{	
			int j;
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// train k*(k-1)/2 models
		
		bool *nonzero=Malloc(bool,l);
		for(i=0;i<l;i++)
			nonzero[i]=false;
		decision_function *f=Malloc(decision_function,nr_class*(nr_class-1)/2);

		double *probA=NULL,*probB=NULL;
		if (param->probability)
		{
			probA=Malloc(double,nr_class*(nr_class-1)/2);
			probB=Malloc(double,nr_class*(nr_class-1)/2);
		}

		int p=0;
		model->param=new svm_parameter[nr_class*(nr_class-1)/2];
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				svm_problem sub_prob;
				int si=start[i], sj=start[j];
				int ci=count[i], cj=count[j];
				sub_prob.l=ci+cj;
				sub_prob.x=Malloc(svm_node *,sub_prob.l);
				sub_prob.y=Malloc(double,sub_prob.l);
				int k;
				for(k=0;k<ci;k++)
				{
					sub_prob.x[k]=x[si+k];
					sub_prob.y[k]=+1;
				}
				for(k=0;k<cj;k++)
				{
					sub_prob.x[ci+k]=x[sj+k];
					sub_prob.y[ci+k]=-1;
				}

				if(param->probability)
					svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);

				f[p]=svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
				copy_param(&model->param[p],param);
				for(k=0;k<ci;k++)
					if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
						nonzero[si+k]=true;
				for(k=0;k<cj;k++)
					if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
						nonzero[sj+k]=true;
				free(sub_prob.x);
				free(sub_prob.y);
				++p;
			}

		// build output

		model->nr_class=nr_class;
		
		model->label=Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model->label[i]=label[i];
		
		model->rho=Malloc(double,nr_class*(nr_class-1)/2);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			model->rho[i]=f[i].rho;

		if(param->probability)
		{
			model->probA=Malloc(double,nr_class*(nr_class-1)/2);
			model->probB=Malloc(double,nr_class*(nr_class-1)/2);
			for(i=0;i<nr_class*(nr_class-1)/2;i++)
			{
				model->probA[i]=probA[i];
				model->probB[i]=probB[i];
			}
		}
		else
		{
			model->probA=NULL;
			model->probB=NULL;
		}

		int total_sv=0;
		int *nz_count=Malloc(int,nr_class);
		model->nSV=Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
		{
			int nSV=0;
			for(int j=0;j<count[i];j++)
				if(nonzero[start[i]+j])
				{	
					++nSV;
					++total_sv;
				}
			model->nSV[i]=nSV;
			nz_count[i]=nSV;
		}
		
		info("Total nSV=%d\n",total_sv);

		model->l=total_sv;
		model->SV=Malloc(svm_node *,total_sv);
		p=0;
		for(i=0;i<l;i++)
			if(nonzero[i]) model->SV[p++]=x[i];

		int *nz_start=Malloc(int,nr_class);
		nz_start[0]=0;
		for(i=1;i<nr_class;i++)
			nz_start[i]=nz_start[i-1]+nz_count[i-1];

		model->sv_coef=Malloc(double *,nr_class-1);
		for(i=0;i<nr_class-1;i++)
			model->sv_coef[i]=Malloc(double,total_sv);

		p=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si=start[i];
				int sj=start[j];
				int ci=count[i];
				int cj=count[j];
				
				int q=nz_start[i];
				int k;
				for(k=0;k<ci;k++)
					if(nonzero[si+k])
						model->sv_coef[j-1][q++]=f[p].alpha[k];
				q=nz_start[j];
				for(k=0;k<cj;k++)
					if(nonzero[sj+k])
						model->sv_coef[i][q++]=f[p].alpha[ci+k];
				++p;
			}

		free(label);
		free(probA);
		free(probB);
		free(count);
		free(perm);
		free(start);
		free(x);
		free(weighted_C);
		free(nonzero);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			free(f[i].alpha);
		free(f);
		free(nz_count);
		free(nz_start);
	}
	
	return model;
}

// Stratified cross validation
void svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start=Malloc(int,nr_fold+1);
	int l=prob->l;
	int *perm=Malloc(int,l);
	int nr_class;

	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	if((param->svm_type == C_SVC ||
	    param->svm_type == NU_SVC) && nr_fold < l)
	{
		int *start=NULL;
		int *label=NULL;
		int *count=NULL;
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

		// random shuffle and then data grouped by fold using the array perm
		int *fold_count=Malloc(int,nr_fold);
		int c;
		int *index=Malloc(int,l);
		for(i=0;i<l;i++)
			index[i]=perm[i];
		for (c=0; c<nr_class; c++) 
			for(i=0;i<count[c];i++)
			{
				int j=i+rand()%(count[c]-i);
				swap(index[start[c]+j],index[start[c]+i]);
			}
		for(i=0;i<nr_fold;i++)
		{
			fold_count[i]=0;
			for (c=0; c<nr_class;c++)
				fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
		}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i]=fold_start[i-1]+fold_count[i-1];
		for (c=0; c<nr_class;c++)
			for(i=0;i<nr_fold;i++)
			{
				int begin=start[c]+i*count[c]/nr_fold;
				int end=start[c]+(i+1)*count[c]/nr_fold;
				for(int j=begin;j<end;j++)
				{
					perm[fold_start[i]]=index[j];
					fold_start[i]++;
				}
			}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i]=fold_start[i-1]+fold_count[i-1];
		free(start);	
		free(label);
		free(count);	
		free(index);
		free(fold_count);
	}
	else
	{
		for(i=0;i<l;i++) perm[i]=i;
		for(i=0;i<l;i++)
		{
			int j=i+rand()%(l-i);
			swap(perm[i],perm[j]);
		}
		for(i=0;i<=nr_fold;i++)
			fold_start[i]=i*l/nr_fold;
	}

	for(i=0;i<nr_fold;i++)
	{
		int begin=fold_start[i];
		int end=fold_start[i+1];
		int j,k;
		struct svm_problem subprob;

		subprob.l=l-(end-begin);
		subprob.x=Malloc(struct svm_node*,subprob.l);
		subprob.y=Malloc(double,subprob.l);
			
		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k]=prob->x[perm[j]];
			subprob.y[k]=prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k]=prob->x[perm[j]];
			subprob.y[k]=prob->y[perm[j]];
			++k;
		}
		struct svm_model *submodel=svm_train(&subprob,param);
		if(param->probability && 
		   (param->svm_type == C_SVC || param->svm_type == NU_SVC))
		{
			double *prob_estimates=Malloc(double,svm_get_nr_class(submodel));
			for(j=begin;j<end;j++)
				target[perm[j]]=svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
			free(prob_estimates);			
		}
		else
			for(j=begin;j<end;j++)
				target[perm[j]]=svm_predict(submodel,prob->x[perm[j]]);
		svm_destroy_model(submodel);
		free(subprob.x);
		free(subprob.y);
	}		
	free(fold_start);
	free(perm);	
}


int svm_get_svm_type(const svm_model *model)
{
	return model->param[0].svm_type;
}

int svm_get_nr_class(const svm_model *model)
{
	return model->nr_class;
}

void svm_get_labels(const svm_model *model, int* label)
{
	if (model->label != NULL)
		for(int i=0;i<model->nr_class;i++)
			label[i]=model->label[i];
}

double svm_get_svr_probability(const svm_model *model)
{
	if ((model->param[0].svm_type == EPSILON_SVR || model->param[0].svm_type == NU_SVR) &&
	    model->probA!=NULL)
		return model->probA[0];
	else
	{
		fprintf(stderr,"Model doesn't contain information for SVR probability inference\n");
		return 0;
	}
}

void svm_predict_values(const svm_model *model, const svm_node *x, double *dec_values)
{
	if(model->param[0].svm_type == ONE_CLASS ||
	   model->param[0].svm_type == EPSILON_SVR ||
	   model->param[0].svm_type == NU_SVR)
	{
		double *sv_coef=model->sv_coef[0];
		double sum=0;
		for(int i=0;i<model->l;i++)
			sum+=sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param[0]);
		sum -= model->rho[0];
		*dec_values=sum;
	}
	else
	{
		int i;
		int nr_class=model->nr_class;
		int l=model->l;
		int p;
		
		double *kvalue=Malloc(double,l*nr_class*(nr_class-1)/2);
		for(p=0;p<nr_class*(nr_class-1)/2;p++)
			for(i=0;i<l;i++)
				kvalue[p*l+i]=Kernel::k_function(x,model->SV[i],model->param[p]);

		int *start=Malloc(int,nr_class);
		start[0]=0;
		for(i=1;i<nr_class;i++)
			start[i]=start[i-1]+model->nSV[i-1];

		p=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				double sum=0;
				int si=start[i];
				int sj=start[j];
				int ci=model->nSV[i];
				int cj=model->nSV[j];
				
				int k;
				double *coef1=model->sv_coef[j-1];
				double *coef2=model->sv_coef[i];
				for(k=0;k<ci;k++)
					sum+=coef1[si+k] * kvalue[p*l+si+k];
				for(k=0;k<cj;k++)
					sum+=coef2[sj+k] * kvalue[p*l+sj+k];
				sum -= model->rho[p];
				dec_values[p]=sum;
				p++;
			}

		free(kvalue);
		free(start);
	}
}

double svm_predict(const svm_model *model, const svm_node *x)
{
	if(model->param[0].svm_type == ONE_CLASS ||
	   model->param[0].svm_type == EPSILON_SVR ||
	   model->param[0].svm_type == NU_SVR)
	{
		double res;
		svm_predict_values(model, x, &res);
		
		if(model->param[0].svm_type == ONE_CLASS)
			return (res>0)?1:-1;
		else
			return res;
	}
	else
	{
		int i;
		int nr_class=model->nr_class;
		double *dec_values=Malloc(double, nr_class*(nr_class-1)/2);
		svm_predict_values(model, x, dec_values);

		int *vote=Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			vote[i]=0;
		int pos=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				if(dec_values[pos++] > 0)
					++vote[i];
				else
					++vote[j];
			}

		int vote_max_idx=0;
		for(i=1;i<nr_class;i++)
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx=i;
		free(vote);
		free(dec_values);
		return model->label[vote_max_idx];
	}
}

double svm_predict_probability(
	const svm_model *model, const svm_node *x, double *prob_estimates)
{
	if ((model->param[0].svm_type == C_SVC || model->param[0].svm_type == NU_SVC) &&
	    model->probA!=NULL && model->probB!=NULL)
	{
		int i;
		int nr_class=model->nr_class;
		double *dec_values=Malloc(double, nr_class*(nr_class-1)/2);
		svm_predict_values(model, x, dec_values);

		double min_prob=1e-7;
		double **pairwise_prob=Malloc(double *,nr_class);
		for(i=0;i<nr_class;i++)
			pairwise_prob[i]=Malloc(double,nr_class);
		int k=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				pairwise_prob[i][j]=min(max(sigmoid_predict(dec_values[k],model->probA[k],model->probB[k]),min_prob),1-min_prob);
				pairwise_prob[j][i]=1-pairwise_prob[i][j];
				k++;
			}
		multiclass_probability(nr_class,pairwise_prob,prob_estimates);

		int prob_max_idx=0;
		for(i=1;i<nr_class;i++)
			if(prob_estimates[i] > prob_estimates[prob_max_idx])
				prob_max_idx=i;
		for(i=0;i<nr_class;i++)
			free(pairwise_prob[i]);
		free(dec_values);
		free(pairwise_prob);	     
		return model->label[prob_max_idx];
	}
	else 
		return svm_predict(model, x);
}

static const char *svm_type_table[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

int read_precomputed(char *precomputed_filename,struct kernel *k)
{
	FILE *fp=fopen(precomputed_filename,"r");
	if(fp==NULL || k==NULL) return 1;

	int num_rows=-1,num_cols=-1;
	fscanf(fp,"%d%d",&num_rows,&num_cols);
	if(num_rows<0 || num_cols<0) return 1;

	k->precomputed_numrows=num_rows;
	k->precomputed_numcols=num_cols;

	k->precomputed=new double*[num_rows];
	for(int row=0;row<num_rows;row++)
		k->precomputed[row]=new double[num_cols];

	for(int row=0;row<num_rows;row++)
		for(int col=0;col<num_cols;col++)
			if(fscanf(fp,"%lf",&k->precomputed[row][col])!=1)
			{
				fprintf(stderr,"Cannot read precomputed kernel value row %d column %d from %s\n",
					row+1,col+1,precomputed_filename);
				for(row=0;row<num_rows;row++)
					delete[] k->precomputed[row];
				delete[] k->precomputed;
				fclose(fp);
				return 1;
			}

	fclose(fp);
	return 0;
}

int read_kernels(FILE *fp,struct svm_parameter *param,int read_num_kernels)
{
	static char buffer[256],buffer2[256],precomputed_filename[256];
	char *s;
	double n,uniform_weight;

	if(fp==NULL) return 1;

	if(read_num_kernels)
	{
		fgets(buffer,256,fp);
		sscanf(buffer,"%d",&param->num_kernels);
	}
	if(param->num_kernels<=0) return 2;

  param->kernels=new kernel[param->num_kernels];

	if(param->kernels==NULL) return 3;

	uniform_weight=1.0/param->num_kernels;
	
	for(int i=0;i<param->num_kernels;i++)
	{
		if(fgets(buffer,256,fp)==NULL)
		{
			delete[] param->kernels;
			return 2;
		}
	
		param->kernels[i].coef=uniform_weight;
		param->kernels[i].kernel_type=RBF;
		param->kernels[i].scale_factor=1.0;
		param->kernels[i].degree=3;
		param->kernels[i].gamma=0;
		param->kernels[i].coef0=0;
		param->kernels[i].precomputed=NULL;
		param->kernels[i].precomputed_filename=NULL;
		s=buffer;

		while(*s!='\0')
		{
			sscanf(s,"%s",buffer2);
			if(buffer2[0]!='-')
			{
				delete[] param->kernels;
				return 2;
			}
	
			while(*s!=' ' && *s!='\t' && *s!='\0') s++;
			while(*s==' ' || *s=='\t') s++;
			if(buffer2[1]=='f')
				sscanf(s,"%255s",precomputed_filename);
			else
				sscanf(s,"%lf",&n);
			switch(buffer2[1])
			{
				case 't':
					param->kernels[i].kernel_type=(int)n;
					break;
				case 'd':
					param->kernels[i].degree=(int)n;
					break;
				case 'g':
					param->kernels[i].gamma=n;
					break;
				case 'r':
					param->kernels[i].coef0=n;
					break;
        case 'w':
          param->kernels[i].coef=(double)n;
        case 'u':
          param->kernels[i].scale_factor=(double)n;
          break;
				case 'f':
					if(param->kernels[i].kernel_type!=PRECOMPUTED || 
						read_precomputed(precomputed_filename,&param->kernels[i])!=0)
					{
						delete[] param->kernels;
						return 2;
					}
					else
					{
						param->kernels[i].precomputed_filename=new char[strlen(precomputed_filename)+1];
						strcpy(param->kernels[i].precomputed_filename,precomputed_filename);
					}
					break;
				default:
					delete[] param->kernels;
					return 2;
			}
			while(*s!=' ' && *s!='\t' && *s!='\0') s++;
			while(*s==' ' || *s=='\t') s++;
		}
	}
	return 0;
}

void save_kernels(FILE *fp,const struct svm_parameter *param)
{
	// remove kernels with very low coefficient
	double sum_d=1.0;
	int num_kernels=0;
  for(int i=0;i<param->num_kernels;i++)
    if(param->kernels[i].coef < DTAU)
      param->kernels[i].coef=0.0;
    else
    {
      sum_d+=param->kernels[i].coef;
      num_kernels++;
    }
	
	fprintf(fp,"%d\n",num_kernels);
	
  // svnvish: Normalize only if we are projecting onto the simplex 
  if(param->d_proj == SIMPLEX)
    sum_d=1.0/sum_d;
  else
    sum_d=1.0;
  
	for(int i=0;i<param->num_kernels;i++)
	{
		if(param->kernels[i].coef < DTAU) continue;
		param->kernels[i].coef*=sum_d;
		if(param->kernels[i].kernel_type == PRECOMPUTED)
			fprintf(fp,"-t %d -w %.17g -u %.17g -f %s.test\n",PRECOMPUTED,param->kernels[i].coef,param->kernels[i].scale_factor,
			param->kernels[i].precomputed_filename);
		else
			fprintf(fp,"-t %d -w %.17g -u %.17g -d %d -g %.17g -r %.17g\n",param->kernels[i].kernel_type,
				param->kernels[i].coef,param->kernels[i].scale_factor,param->kernels[i].degree,param->kernels[i].gamma,
				param->kernels[i].coef0);
	}
}

int svm_save_model(const char *model_file_name, const svm_model *model)
{
	FILE *fp=fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	fprintf(fp,"svm_type %s\n", svm_type_table[model->param[0].svm_type]);

	int nr_class=model->nr_class;
	fprintf(fp, "nr_class %d\n", nr_class);

	if(model->param[0].svm_type == ONE_CLASS ||
	   model->param[0].svm_type == EPSILON_SVR ||
	   model->param[0].svm_type == NU_SVR)
	{
		fprintf(fp,"num_kernels ");
		save_kernels(fp,model->param);
	}
	else
	{
		int p=0;
		for(int i=0;i<model->nr_class;i++)
			for(int j=i+1;j<model->nr_class;j++)
			{
				fprintf(fp,"num_kernels ");
				save_kernels(fp,&model->param[p]);
				p++;
			}
	}

	int l=model->l;
	fprintf(fp, "total_sv %d\n",l);
	
	{
		fprintf(fp, "rho");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %.17g",model->rho[i]);
		fprintf(fp, "\n");
	}
	
	if(model->label)
	{
		fprintf(fp, "label");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->label[i]);
		fprintf(fp, "\n");
	}

	if(model->probA) // regression has probA only
	{
		fprintf(fp, "probA");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %.17g",model->probA[i]);
		fprintf(fp, "\n");
	}
	if(model->probB)
	{
		fprintf(fp, "probB");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %.17g",model->probB[i]);
		fprintf(fp, "\n");
	}

	if(model->nSV)
	{
		fprintf(fp, "nr_sv");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->nSV[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "SV\n");
	const double * const *sv_coef=model->sv_coef;
	const svm_node * const *SV=model->SV;

	for(int i=0;i<l;i++)
	{
		for(int j=0;j<nr_class-1;j++)
			fprintf(fp, "%.17g ",sv_coef[j][i]);

		const svm_node *p=SV[i];

		while(p->index != -1)
		{
			fprintf(fp,"%d:%.8g ",p->index,p->value);
			p++;
		}
		fprintf(fp, "\n");
	}
	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

static char *line=NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line=(char *) realloc(line,max_line_len);
		len=(int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

svm_model *svm_load_model(const char *model_file_name)
{
	FILE *fp=fopen(model_file_name,"rb");
	if(fp==NULL) return NULL;
	
	// read parameters

	svm_model *model=Malloc(svm_model,1);
	model->param=NULL;
	model->rho=NULL;
	model->probA=NULL;
	model->probB=NULL;
	model->label=NULL;
	model->nSV=NULL;

	char cmd[81];
	int pn=0,svm_type=-1;
	while(1)
	{
		fscanf(fp,"%80s",cmd);
    
		if(strcmp(cmd,"svm_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;svm_type_table[i];i++)
			{
				if(strcmp(svm_type_table[i],cmd)==0)
				{
					svm_type=i;
					break;
				}
			}
			if(svm_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown svm type.\n");
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		}
		else if(strcmp(cmd,"num_kernels")==0)
		{
			if(model->param==NULL || read_kernels(fp,&model->param[pn],1)!=0)
			{
				free(model->rho);
				free(model->label);
				free(model->nSV);
				delete model->param;
				free(model);
				return NULL;
			}
			pn++;
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&model->nr_class);
			if(svm_type == -1)
			{
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
			if(svm_type == ONE_CLASS ||
			   svm_type == EPSILON_SVR ||
			   svm_type == NU_SVR)
			{
				model->param=new svm_parameter[1];
				model->param[0].svm_type=svm_type;
				model->param[0].num_kernels=-1;
			}
			else
			{
				model->param=new svm_parameter[model->nr_class*(model->nr_class-1)/2];
				for(int i=0;i<model->nr_class*(model->nr_class-1)/2;i++)
				{
					model->param[i].svm_type=svm_type;
					model->param[i].kernels=NULL;
					model->param[i].num_kernels=-1;
				}
			}
		}
		else if(strcmp(cmd,"total_sv")==0)
			fscanf(fp,"%d",&model->l);
		else if(strcmp(cmd,"rho")==0)
		{
			int n=model->nr_class * (model->nr_class-1)/2;
			model->rho=Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->rho[i]);
		}
		else if(strcmp(cmd,"label")==0)
		{
			int n=model->nr_class;
			model->label=Malloc(int,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&model->label[i]);
		}
		else if(strcmp(cmd,"probA")==0)
		{
			int n=model->nr_class * (model->nr_class-1)/2;
			model->probA=Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->probA[i]);
		}
		else if(strcmp(cmd,"probB")==0)
		{
			int n=model->nr_class * (model->nr_class-1)/2;
			model->probB=Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->probB[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n=model->nr_class;
			model->nSV=Malloc(int,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&model->nSV[i]);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c=getc(fp);
				if(c==EOF || c=='\n') break;	
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free(model->rho);
			free(model->label);
			free(model->nSV);
			free(model);
			return NULL;
		}
	}

	// read sv_coef and SV

	int elements=0;
	long pos=ftell(fp);

	max_line_len=1024;
	line=Malloc(char,max_line_len);
	char *p,*endptr,*idx,*val;

	while(readline(fp)!=NULL)
	{
		p=strtok(line,":");
		while(1)
		{
			p=strtok(NULL,":");
			if(p == NULL)
				break;
			++elements;
		}
	}
	elements+=model->l;

	fseek(fp,pos,SEEK_SET);

	int m=model->nr_class - 1;
	int l=model->l;
	model->sv_coef=Malloc(double *,m);
	int i;
	for(i=0;i<m;i++)
		model->sv_coef[i]=Malloc(double,l);
	model->SV=Malloc(svm_node*,l);
	svm_node *x_space=NULL;
	if(l>0) x_space=Malloc(svm_node,elements+l);

	int j=0;
	for(i=0;i<l;i++)
	{
		readline(fp);
		model->SV[i]=&x_space[j];

		p=strtok(line, " \t");
		model->sv_coef[0][i]=strtod(p,&endptr);
		for(int k=1;k<m;k++)
		{
			p=strtok(NULL, " \t");
			model->sv_coef[k][i]=strtod(p,&endptr);
		}

		while(1)
		{
			idx=strtok(NULL, ":");
			val=strtok(NULL, " \t");

			if(val == NULL)
				break;
			x_space[j].index=(int) strtol(idx,&endptr,10);
			x_space[j].value=strtod(val,&endptr);

			++j;
		}
		x_space[j++].index=-1;
	}
	free(line);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	model->free_sv=1;	// XXX
	return model;
}

void svm_destroy_model(svm_model* model)
{
	if(model->free_sv && model->l > 0)
		free((void *)(model->SV[0]));
	for(int i=0;i<model->nr_class-1;i++)
		free(model->sv_coef[i]);
	free(model->SV);
	free(model->sv_coef);
	free(model->rho);
	free(model->label);
	free(model->probA);
	free(model->probB);
	free(model->nSV);
	if(model->param[0].svm_type != ONE_CLASS &&
	   model->param[0].svm_type != EPSILON_SVR &&
	   model->param[0].svm_type != NU_SVR)
	{
		int p=0;
		for(int i=0;i<model->nr_class;i++)
			for(int j=i+1;j<model->nr_class;j++)
			{
				svm_destroy_param(&model->param[p]);
				p++;
			}
	}
	else
		svm_destroy_param(model->param);
	delete[] model->param;
	free(model);
}

void svm_destroy_param(svm_parameter* param)
{
	if(param==NULL) return;
	free(param->weight_label);
	free(param->weight);
	if(param->kernels!=NULL)
		for(int i=0;i<param->num_kernels;i++)
			if(param->kernels[i].kernel_type==PRECOMPUTED)
			{
				for(int row=0;row<param->kernels[i].precomputed_numrows;row++)
					delete[] param->kernels[i].precomputed[row];
				delete[] param->kernels[i].precomputed;
				delete[] param->kernels[i].precomputed_filename;
			}
	delete[] param->kernels;
}

const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param)
{
	// svm_type

	int svm_type=param->svm_type;
	if(svm_type != C_SVC &&
	   svm_type != NU_SVC &&
	   svm_type != ONE_CLASS &&
	   svm_type != EPSILON_SVR &&
	   svm_type != NU_SVR)
		return "unknown svm type";
	
	// kernel_type, degree
	if(param->num_kernels<=0 || param->kernels==NULL)
		return "no kernel";
	
	for(int i=0;i<param->num_kernels;i++)
	{
		int kernel_type=param->kernels[i].kernel_type;
		if(kernel_type != LINEAR &&
		   kernel_type != POLY &&
		   kernel_type != RBF &&
		   kernel_type != SIGMOID &&
		   kernel_type != PRECOMPUTED)
			return "unknown kernel type";

		if(kernel_type == PRECOMPUTED)
		{
			if(prob->l != param->kernels[i].precomputed_numrows)
				return "number of precomputed kernel rows should be the "
				"same as number of data points";
			if(prob->l != param->kernels[i].precomputed_numcols)
				return "number of precomputed kernel columns should be the "
				"same as number of data points";
		}

		if(param->kernels[i].gamma < 0)
			return "gamma < 0";

		if(param->kernels[i].degree < 0)
			return "degree of polynomial kernel < 0";
	}

  double sum_d=0.0;
	for(int i=0;i<param->num_kernels;i++)
    sum_d+=param->kernels[i].coef;
  
  // Sum may be slightly off from one 
  if(sum_d > 1.0+TAU || sum_d < 1.0-TAU)
  {
    fprintf(stderr, "%f", sum_d);
    return "kernel coefficient sum != 1.0";
  }
  
	// cache_size,eps,C,nu,p,shrinking

	if(param->cache_size <= 0)
		return "cache_size <= 0";

	if(param->eps <= 0)
		return "eps <= 0";

	if(svm_type == C_SVC ||
	   svm_type == EPSILON_SVR ||
	   svm_type == NU_SVR)
		if(param->C <= 0)
			return "C <= 0";

	if(svm_type == NU_SVC ||
	   svm_type == ONE_CLASS ||
	   svm_type == NU_SVR)
		if(param->nu <= 0 || param->nu > 1)
			return "nu <= 0 or nu > 1";

	if(svm_type == EPSILON_SVR)
		if(param->p < 0)
			return "p < 0";

	if(param->shrinking != 0 &&
	   param->shrinking != 1)
		return "shrinking != 0 and shrinking != 1";

	if(param->probability != 0 &&
	   param->probability != 1)
		return "probability != 0 and probability != 1";

	if(param->probability == 1 &&
	   svm_type == ONE_CLASS)
		return "one-class SVM probability output not supported yet";

  if(param->solver_type != SMO &&
     param->solver_type != SPG &&
     param->solver_type != SPGF &&
     param->solver_type != PGD &&
     param->solver_type != SMKL &&
     param->solver_type != MD)
    return "unknown solver";
  
  if(param->d_regularizer != ENT &&
     param->d_regularizer != L1 &&
     param->d_regularizer != L2 &&
     param->d_regularizer != LP)
    return "unknown regularizer";

  if(param->d_regularizer == LP && 
     (param->L_p < 1.0))
    return "p for Lp regularization must be >= 1.0";
  
  if(param->d_proj != SIMPLEX &&
     param->d_proj != NN_ORTHANT)
    return "unknown projection";
  
  if((param->solver_type == SMO || param->solver_type == SPG || param->solver_type == SPGF || param->solver_type == PGD) && param->lambda<=0.0)
    return "lambda must be > 0 for the coordinate descent solver";
  
  if(param->solver_type == MD && param->d_proj!=SIMPLEX)
    return "Mirror Descent solver is only compatible with simplex constraints";

  if(param->d_regularizer == ENT && param->d_proj != SIMPLEX)
    return "Entropic regularization is only compatible with simplex constraints";
  
  if(param->d_regularizer == ENT &&
     param->solver_type == SMO &&
     (param->obj_threshold<=0.0 || param->obj_threshold>0.5))
    return "obj_threshold must be in (0, 0.5]";
  
  if(param->d_regularizer == ENT &&
     param->solver_type == SMO &&
     param->diff_threshold<=0.0)
    return "diff_threshold must be > 0";
  
  if((param->d_regularizer == L1 && 
     param->solver_type == SMO) || (param->d_regularizer == LP && param->L_p==1.0 && 
     param->solver_type == SMO))
    return "SMO solver for L1 regularizer not supported yet. Use Lp with p=1.01 to get sparse solution. \n You can also use SPG solver, which is supported for L1 regularizer.";
  
  if(param->d_regularizer == L2 &&
     param->d_proj != NN_ORTHANT && 
     param->solver_type == SMO)
    return "SMO solver for L2 regularizer only works with nn-orthant.";

  if(param->d_regularizer == LP &&
     param->d_proj != NN_ORTHANT && 
     param->solver_type == SMO)
    return "SMO solver for Lp regularizer only works with nn-orthant.";

  if(param->d_regularizer == L1 &&
     param->d_proj != NN_ORTHANT && 
     param->solver_type == SPG)
    return "SPG solver for L1 regularizer only works with nn-orthant.";

  if(param->d_regularizer == L2 &&
     param->d_proj != NN_ORTHANT && 
     param->solver_type == SPG)
    return "SPG solver for L2 regularizer only works with nn-orthant.";

  if(param->d_regularizer == LP &&
     param->d_proj != NN_ORTHANT && 
     param->solver_type == SPG)
    return "SPG solver for Lp regularizer only works with nn-orthant.";
    
  if(param->d_regularizer == L1 &&
     param->d_proj != NN_ORTHANT && 
     param->solver_type == PGD)
    return "PGD solver for L1 regularizer only works with nn-orthant.";

  if(param->d_regularizer == L2 &&
     param->d_proj != NN_ORTHANT && 
     param->solver_type == PGD)
    return "PGD solver for L2 regularizer only works with nn-orthant.";

  if(param->d_regularizer == LP &&
     param->d_proj != NN_ORTHANT && 
     param->solver_type == PGD)
    return "PGD solver for Lp regularizer only works with nn-orthant.";    

  
	// check whether nu-svc is feasible
	
	if(svm_type == NU_SVC)
	{
		int l=prob->l;
		int max_nr_class=16;
		int nr_class=0;
		int *label=Malloc(int,max_nr_class);
		int *count=Malloc(int,max_nr_class);

		int i;
		for(i=0;i<l;i++)
		{
			int this_label=(int)prob->y[i];
			int j;
			for(j=0;j<nr_class;j++)
				if(this_label == label[j])
				{
					++count[j];
					break;
				}
			if(j == nr_class)
			{
				if(nr_class == max_nr_class)
				{
					max_nr_class *= 2;
					label=(int *)realloc(label,max_nr_class*sizeof(int));
					count=(int *)realloc(count,max_nr_class*sizeof(int));
				}
				label[nr_class]=this_label;
				count[nr_class]=1;
				++nr_class;
			}
		}
	
		for(i=0;i<nr_class;i++)
		{
			int n1=count[i];
			for(int j=i+1;j<nr_class;j++)
			{
				int n2=count[j];
				if(param->nu*(n1+n2)/2 > min(n1,n2))
				{
					free(label);
					free(count);
					return "specified nu is infeasible";
				}
			}
		}
		free(label);
		free(count);
	}

	return NULL;
}

int svm_check_probability_model(const svm_model *model)
{
	return ((model->param[0].svm_type == C_SVC || model->param[0].svm_type == NU_SVC) &&
		model->probA!=NULL && model->probB!=NULL) ||
		((model->param[0].svm_type == EPSILON_SVR || model->param[0].svm_type == NU_SVR) &&
		 model->probA!=NULL);
}

//==========STARTS QUARTIC CODE==========

/* compute real roots of cubic polynomial with real coefficients
 * (c) 2008, 2009 Nicol N. Schraudolph  - all rights reserved -
 * see Wikipedia article on Cubic_equation
 */

inline int linear(double A, double B,double *x)
/* returns the number n of real roots of the polynomial
 *     A*x + B=0
 * side effect: stores the real roots in x[0]..x[n-1]
 */
{
    if (A == 0.0)
    {
        if (B != 0) return 0;
        else
        {
            fprintf(stderr, "degenerate: all coefficients zero!\n");
            return -1;
        }
    }
    x[0]=-B/A;
    return 1;
}

int quadratic(double A, double B, double C,double *x)
/* returns the number n of real roots of the polynomial
 *     A*x^2 + B*x + C=0
 * side effect: stores the real roots in x[0]..x[n-1]
 */
{
    if (A == 0.0) return linear(B, C, x);
    const double det=B*B - 4.0*A*C;
    int n=0;

    if (det >= 0.0)
    {
        x[n]=sqrt(det);
        if (det > 0.0)  // two distinct roots
        {
            x[n+1]=-x[n];
            x[n] -= B;
            x[n] /= 2.0*A;
            ++n;
        }
        x[n] -= B;
        x[n] /= 2.0*A;
        ++n;
    }

    return n;
}

int cubic(double A, double B, double C, double D,double *x)
/* returns the number n of real roots of the polynomial
 *     A*x^3 + B*x^2 + C*x + D=0
 * side effect: stores the real roots in x[0]..x[n-1]
 */
{
    if (A == 0.0) return quadratic(B, C, D, x);
    int n=0;

    const double A3=(1.0/3.0)/A;
    const double BA3=B*A3;
    const double q=C*A3 - BA3*BA3;
    const double r=(BA3*C - D)/(2.0*A) - BA3*BA3*BA3;
    const double det=q*q*q + r*r;

    if (det > 0.0)  // single real root
    {
        const double s=sqrt(det);
        x[n]=cbrt(r + s) + cbrt(r - s) - BA3;
        ++n;
    }
    else if (det == 0.0)  // single + double real root
    {
        const double s=cbrt(r);
        x[n]=2.0*s - BA3;
        ++n;
        if (s > 0.0)  // not a triple root
        {
            x[n]=-s - BA3;
            ++n;
        }
    }
    else  // 3 distinct real roots
    {
        const double rho=cbrt(sqrt(r*r - det));
        const double theta=atan2(sqrt(-det), r)/3.0;
        const double spt=rho*cos(theta);
        const double smt=rho*sin(theta)*sqrt(3.0);

        x[n]=2.0*spt - BA3;
        ++n;
        x[n]=-spt - BA3;
        x[n+1]=x[n];
        x[n]+=smt;
        ++n;
        x[n] -= smt;
        ++n;
    }

    return n;
}
