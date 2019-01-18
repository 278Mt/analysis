#include<iostream>
#include<cmath>
#include<fstream>
#include<Eigen/Dense>
#include<Eigen/Core>
#include<Eigen/LU>

// g++ lda_test.cpp -I /usr/local/include/eigen3/

using namespace Eigen;
using namespace std;

int data = 100;
int buf_size = 16;

MatrixXd _class_aves(MatrixXd X, VectorXd Y);
MatrixXd _class_cov(MatrixXd X, VectorXd Y);
VectorXd _ave(MatrixXd X);
MatrixXd _cov(MatrixXd X);
double norm(VectorXd v);
VectorXd veclog(VectorXd priors);
VectorXd diag(MatrixXd m);
void _solve_eigen(MatrixXd X, VectorXd Y, MatrixXd *coef, VectorXd *intercept);
void fit(MatrixXd X, VectorXd Y, VectorXd *coef, double *intercept);
void _init(MatrixXd *X, VectorXd *Y);

int main(int argc, char **argv) {

    MatrixXd X(data, buf_size);
    VectorXd Y(data);
    _init(&X, &Y);

    VectorXd coef(buf_size);
    double intercept;
    fit(X, Y, &coef, &intercept);

    cout << "coef = " << endl << coef << endl;
    cout << "intercept = " << endl << intercept << endl;

    return 0;
}

MatrixXd _class_aves(MatrixXd X, VectorXd Y) {
    MatrixXd aves(2, buf_size); aves = MatrixXd::Zero(2,buf_size);

    for(int i=0; i<data; i+=1) {
        if(Y(i) == 0) {
            for(int j=0; j<buf_size; j+=1) {
                aves(0,j) += X(i, j);
            }
        }
        else if(Y(i) == 1) {
            for(int j=0; j<buf_size; j+=1) {
                aves(1,j) += X(i, j);
            }
        }
    }
    aves /= (data / 2);

    return aves;
}

MatrixXd _class_cov(MatrixXd X, VectorXd Y) {

    MatrixXd cov(buf_size,buf_size); cov = MatrixXd::Zero(buf_size,buf_size);
    MatrixXd aves(2,buf_size); aves = _class_aves(X, Y);
    for(int i=0; i<buf_size; i+=1) {
        for(int j=0; j<buf_size; j+=1) {
            for(int k=0; k<data/2; k+=1) {
                cov(i,j) += (X(k,i) - aves(0,i)) * (X(k,j) - aves(0,j)) + (X(k+data/2,i) - aves(1,i)) * (X(k+data/2,j) - aves(1,j));
            }
        }
    }
    cov /= data;

    return cov;
}

VectorXd _ave(MatrixXd X) {
    VectorXd ave(buf_size); ave = VectorXd::Zero(buf_size);

    for(int i=0; i<data; i+=1) {
        for(int j=0; j<buf_size; j+=1) {
            ave(j) += X(i, j);
        }
    }
    ave /= data;

    return ave;
}

MatrixXd _cov(MatrixXd X) {
    MatrixXd cov(buf_size,buf_size); cov = MatrixXd::Zero(buf_size,buf_size);
    VectorXd ave(buf_size); ave = _ave(X);
    for(int i=0; i<buf_size; i+=1) {
        for(int j=0; j<buf_size; j+=1) {
            for(int k=0; k<data; k+=1) {
                cov(i,j) += (X(k,i) - ave(i)) * (X(k,j) - ave(j));
            }
        }
    }
    cov /= data;
    return cov;
}

double norm(VectorXd v) {
    double res = 0;
    for(int i=0; i<v.rows(); i+=1) {
        res += pow(v(i),2);
    }
    return sqrt(res);
}

VectorXd veclog(VectorXd priors) {
    VectorXd res(priors.rows());
    for(int i=0; i<priors.rows(); i+=1) {
        res(i) = log(priors(i));
    }
    return res;
}

VectorXd diag(MatrixXd m) {
    VectorXd res(m.cols());
    for(int i=0; i<m.cols(); i+=1) {
        res(i) = m(i,i);
    }
    return res;
}

void _solve_eigen(MatrixXd X, VectorXd Y, MatrixXd *coef, VectorXd *intercept) {

    MatrixXd aves(2,buf_size); aves = _class_aves(X, Y);

    MatrixXd Sw(buf_size,buf_size); Sw = _class_cov(X, Y);
    MatrixXd St(buf_size,buf_size); St = _cov(X);
    MatrixXd Sb(buf_size,buf_size); Sb = St - Sw;

    GeneralizedSelfAdjointEigenSolver<MatrixXd> es(Sb, Sw);

    VectorXd evals(buf_size);          evals = es.eigenvalues();
    MatrixXd evecs(buf_size,buf_size); evecs = es.eigenvectors();

    for(int i=0; i<buf_size; i+=1) {
        evecs.col(i) /= norm(evecs.col(i));
    }

    *coef = aves * evecs * evecs.transpose();
    VectorXd priors(2); priors << 0.5, 0.5;
    *intercept = -0.5 * diag(aves * coef->transpose()) + veclog(priors);
}

void fit(MatrixXd X, VectorXd Y, VectorXd *coef, double *intercept) {
    MatrixXd coef_(2, buf_size);
    VectorXd intercept_(2);

    _solve_eigen(X, Y, &coef_, &intercept_);

    *coef = coef_.row(1) - coef_.row(0);
    *intercept = intercept_(1) - intercept_(0);
}

void _init(MatrixXd *X, VectorXd *Y) {

    ifstream ifs("cat.csv");
    char split;
    string str;
    ifs >> str;
    for(int j=0; j<data; j+=1) {
        for(int i=0; i<buf_size; i+=1) {
            ifs >> (*X)(j,i) >> split;
        }
    }

    *Y = VectorXd::Zero(data);
    for(int i=data/2; i<data; i+=1) {
        (*Y)(i) = 1;
    }
}

