#include <iostream>
#include <math.h>
#include <omp.h>
#include <queue>
#include <stack>
#include <vector>
#include <time.h>
using namespace std;

double f(double x);

int main() {
  double a = 1, b = 100, e = 1e-6, s = 12;
  double M = 0;
  int size = 50000;
  omp_lock_t lock;
  omp_init_lock(&lock);
  double **candidates = new double *[size];
  for (int i = 0; i < size; ++i) {
    candidates[i] = new double [4];
    for (int j = 0; j < 4; ++j) candidates[i][j] = 0;
  }
  int count = 1;
  candidates[0][0] = a;
  candidates[0][1] = b;
  candidates[0][2] = f(a);
  candidates[0][3] = f(b);
  int start = 0;
  bool isFull = false;
  double beginTime = omp_get_wtime();
  // BFS
  while (2 * count < 5000) {
    int newCount = 0;
    int next = (start + count) % size;
    int newStart = next;
    //#pragma omp parallel for schedule(dynamic, 1) reduction(+:newCount)
    for (int i = 0; i < count; ++i) {
      int pos = (start + i) % size;
      omp_set_lock(&lock);
      M = max(M, max(candidates[pos][2], candidates[pos][3]));
      omp_unset_lock(&lock);
      if ((candidates[pos][2]+candidates[pos][3]+s*(candidates[pos][1]-candidates[pos][0]))/2 >= M+e) {
        double mid = (candidates[pos][0] + candidates[pos][1]) / 2;
        double val = f(mid);
        newCount += 2;
        omp_set_lock(&lock);
        candidates[next][0] = candidates[pos][0];
        candidates[next][1] = mid;
        candidates[next][2] = candidates[pos][2];
        candidates[next][3] = val;
        next = (++next) % size;
        candidates[next][0] = mid;
        candidates[next][1] = candidates[pos][1];
        candidates[next][2] = val;
        candidates[next][3] = candidates[pos][3];
        next = (++next) % size;
        omp_unset_lock(&lock);
      }
    }
    count = newCount;
    start = newStart;
  }

  // DFS
  #pragma omp parallel for schedule(dynamic, 100)
  for (int i = 0; i < count; ++i) {
    omp_set_lock(&lock);
    double tmpM = M;
    omp_unset_lock(&lock);
    int pos = ((i + start) % size);
    stack<vector<double> > st;
    vector<double> inner(4, 0);
    inner[0] = candidates[pos][0];
    inner[1] = candidates[pos][1];
    inner[2] = candidates[pos][2];
    inner[3] = candidates[pos][3];
    st.push(inner);
    while (!st.empty()) {
      vector<double> cur = st.top();
      st.pop();
      tmpM = max(tmpM, max(cur[2], cur[3]));
      if ((cur[2]+cur[3]+s*(cur[1]-cur[0]))/2 >= tmpM + e) {
        double mid = (cur[0] + cur[1]) / 2;
        double val = f(mid);
        vector<double> newInner(4, 0);
        newInner[0] = mid;
        newInner[1] = cur[1];
        newInner[2] = val;
        newInner[3] = cur[3];
        st.push(newInner);
        newInner[0] = cur[0];
        newInner[1] = mid;
        newInner[2] = cur[2];
        newInner[3] = val;
        st.push(newInner);
      }
    }
    omp_set_lock(&lock);
    M = tmpM;
    omp_unset_lock(&lock);
  }
  cout << omp_get_wtime() - beginTime << endl;
  omp_destroy_lock(&lock);
  cout.precision(16);
  cout << fixed << M << endl;
  return 0;
}

double f(double x) {
  double sum = 0;
  for (int i = 100; i >= 1; --i) {
    double innerSum = x;
    for (int j = i; j >= i; --j) {
      innerSum += pow(x + 0.5 * j, -3.3);
    }
    sum += sin(innerSum) / pow(1.3, i);
  }
  return sum;
}
