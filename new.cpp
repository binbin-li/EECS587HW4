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
  int size = 5000;
  omp_lock_t lock, Mlock;
  omp_init_lock(&lock);
  omp_init_lock(&Mlock);
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
  while (2 * count < size) {
    int newCount = 0;
    int next = (start + count) % size;
    int newStart = next;
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:newCount)
    for (int i = 0; i < count; ++i) {
      int pos = (start + i) % size;
      double comparedNum = max(candidates[pos][2], candidates[pos][3]);
      omp_set_lock(&Mlock);
      M = max(M, comparedNum);
      omp_unset_lock(&Mlock);
      double left = candidates[pos][0], right = candidates[pos][1];
      double leftVal = candidates[pos][2], rightVal = candidates[pos][3];
      if ((leftVal + rightVal + s*(right - left))/2 >= M+e) {
        double mid = (left + right) / 2;
        double val = f(mid);
        newCount += 2;
        omp_set_lock(&lock);
        candidates[next][0] = left;
        candidates[next][1] = mid;
        candidates[next][2] = leftVal;
        candidates[next][3] = val;
        next = (++next) % size;
        candidates[next][0] = mid;
        candidates[next][1] = right;
        candidates[next][2] = val;
        candidates[next][3] = rightVal;
        next = (++next) % size;
        omp_unset_lock(&lock);
        /*
        if ((leftVal + val + s * (mid - left)) / 2 >= M + e) {
          omp_set_lock(&lock);
          candidates[next][0] = left;
          candidates[next][1] = mid;
          candidates[next][2] = leftVal;
          candidates[next][3] = val;
          next = (++next) % size;
          omp_unset_lock(&lock);
        }
        if ((mid + rightVal + s * (right - mid)) / 2 >= M + e) {
          omp_set_lock(&lock);
          candidates[next][0] = mid;
          candidates[next][1] = right;
          candidates[next][2] = val;
          candidates[next][3] = rightVal;
          next = (++next) % size;
          omp_unset_lock(&lock);
        }
        */
      }
    }
    count = newCount;
    start = newStart;
  }

  cout << omp_get_wtime() - beginTime << endl;

  // DFS
  #pragma omp parallel for schedule(dynamic, 5)
  for (int i = 0; i < count; ++i) {
    omp_set_lock(&Mlock);
    double tmpM = M;
    omp_unset_lock(&Mlock);
    int pos = ((i + start) % size);

    /*
    double pseudoSt[64][4];
    int curPos = 0;
    pseudoSt[0][0] = candidates[pos][0];
    pseudoSt[0][1] = candidates[pos][1];
    pseudoSt[0][2] = candidates[pos][2];
    pseudoSt[0][3] = candidates[pos][3];
    while (curPos >= 0) {
      double left = pseudoSt[curPos][0], right = pseudoSt[curPos][1];
      double leftVal = pseudoSt[curPos][2], rightVal = pseudoSt[curPos][3];
      --curPos;
      tmpM = max(tmpM, max(leftVal, rightVal));
      if ((leftVal + rightVal + s * (right - left)) / 2 >= tmpM + e) {
        double mid = (left + right) / 2;
        double val = f(mid);
        ++curPos;
        pseudoSt[curPos][0] = mid;
        pseudoSt[curPos][1] = right;
        pseudoSt[curPos][2] = val;
        pseudoSt[curPos][3] = rightVal;
        ++curPos;
        pseudoSt[curPos][0] = left;
        pseudoSt[curPos][1] = mid;
        pseudoSt[curPos][2] = leftVal;
        pseudoSt[curPos][3] = val;
        if ((leftVal + val + s * (mid - left)) / 2 >= tmpM + e) {
          ++curPos;
          pseudoSt[curPos][0] = mid;
          pseudoSt[curPos][1] = right;
          pseudoSt[curPos][2] = val;
          pseudoSt[curPos][3] = rightVal;
        }
        if ((mid + rightVal + s * (right - mid)) / 2 >= tmpM + e) {
          ++curPos;
          pseudoSt[curPos][0] = left;
          pseudoSt[curPos][1] = mid;
          pseudoSt[curPos][2] = leftVal;
          pseudoSt[curPos][3] = val;
        }
      }
    }
    */
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
    omp_set_lock(&Mlock);
    M = max(tmpM, M);
    omp_unset_lock(&Mlock);
  }
  cout << omp_get_wtime() - beginTime << endl;
  omp_destroy_lock(&lock);
  omp_destroy_lock(&Mlock);
  cout.precision(16);
  cout << fixed << M << endl;
  return 0;
}

double f(double x) {
  double sum = 0;
  for (int i = 100; i >= 1; --i) {
    double innerSum = x;
    for (int j = i; j >= 1; --j) {
      innerSum += pow(x + 0.5 * j, -3.3);
    }
    sum += sin(innerSum) / pow(1.3, i);
  }
  return sum;
}
