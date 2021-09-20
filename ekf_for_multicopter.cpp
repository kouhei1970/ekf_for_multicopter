/**
 Copyright (c) 2021 Kouhei Ito 
*/

#include <stdio.h>
#include <iostream>
#include <random>
#include "pico/stdlib.h"
#include <Eigen/Dense>
#define GRAV (9.80665)
#define MN (1.0)
#define MD (0.1)

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::Matrix;
using Eigen::PartialPivLU;
using namespace Eigen;

uint8_t rk4(uint8_t (*func)(float t, Matrix<float, 4, 1> quat , Matrix<float, 3, 1>x, Matrix<float, 4, 1> &k),
            float t, 
            float h, 
            Matrix<float, 4 ,1> &y, 
            Matrix<float, 3, 1> &x)
{
  uint8_t ret;
  Matrix<float, 4, 1> k1,k2,k3,k4;
  func(t,       y,            x, k1);
  func(t+0.5*h, y + 0.5*h*k1, x, k2);
  func(t+0.5*h, y + 0.5*h*k2, x, k3);
  func(t+h,     y +     h*k3, x, k4);
  y = y + h*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;
  return 0;
}

uint8_t quatdot(float t, Matrix<float, 4, 1> quat, Matrix<float, 3, 1> omega , Matrix<float, 4, 1> &k)
{
  Matrix4f A;
  float p=omega(0,0);
  float q=omega(1,0);
  float r=omega(2,0);
  A<<0,-0.5*p,-0.5*q,-0.5*r,  0.5*p,0,0.5*r,-0.5*q,  0.5*q,-0.5*r,0,0.5*p,  0.5*r,0.5*q,-0.5*p,0;  
  k=A*quat;
  return 0;
}

uint8_t state_equation(Matrix<float, 7, 1>&x, Matrix<float, 3, 1> omega_m, Matrix<float, 3, 1> beta, float dt)
{
  float q0=x(0, 0);
  float q1=x(1, 0);
  float q2=x(2, 0);
  float q3=x(3, 0);
  float dp=x(4, 0);
  float dq=x(5, 0);
  float dr=x(6, 0);
  float pm=omega_m(0, 0);
  float qm=omega_m(1, 0);
  float rm=omega_m(2, 0);
  float betax=beta(0, 0);
  float betay=beta(1, 0);
  float betaz=beta(2, 0);

  x(0, 0) = q0 + 0.5*(-(pm-dp)*q1 -(qm-dq)*q2 -(rm-dr)*q3)*dt;
  x(1, 0) = q1 + 0.5*( (pm-dp)*q0 +(rm-dr)*q2 -(qm-dq)*q3)*dt;
  x(2, 0) = q2 + 0.5*( (qm-dq)*q0 -(rm-dr)*q2 +(pm-dp)*q3)*dt;
  x(3, 0) = q3 + 0.5*( (rm-dr)*q0 +(qm-dq)*q1 -(pm-dp)*q2)*dt;
  x(4, 0) = dp -betax*dp*dt;
  x(5, 0) = dq -betay*dq*dt;
  x(6, 0) = dr -betaz*dr*dt;

  return 0;
}

uint8_t observation_equation(Matrix<float, 7, 1>x, Matrix<float, 6, 1>&z, float g, float mn, float md){
  float q0 = x(0, 0);
  float q1 = x(1, 0);
  float q2 = x(2, 0);
  float q3 = x(3, 0);

  z(0, 0) = 2.0*(q1*q3 - q0*q2)*g;
  z(1, 0) = 2.0*(q2*q3 + q0*q1)*g;
  z(2, 0) = (q0*q0 - q1*q1 - q2*q2 + q3*q3)*g ;
  z(3, 0) = (q0*q0 + q1*q1 - q2*q2 - q3*q3)*mn + 2.0*(q1*q3 - q0*q2)*md;
  z(4, 0) = 2.0*(q1*q2 - q0*q3)*mn + 2.0*(q2*q3 + q0*q1)*md;
  z(5, 0) = 2.0*(q1*q3 + q0*q2)*mn + (q0*q0 - q1*q1 - q2*q2 + q3*q3)*md;
  
  return 0;
}

uint8_t F_jacobian(Matrix<float, 7, 7>&F, Matrix<float,7, 1> x, Matrix<float, 3, 1> omega, Matrix<float, 3, 1> beta, float dt){
  float q0=x(0, 0);
  float q1=x(1, 0);
  float q2=x(2, 0);
  float q3=x(3, 0);
  float dp=x(4, 0);
  float dq=x(5, 0);
  float dr=x(6, 0);
  float pm=omega(0, 0);
  float qm=omega(1, 0);
  float rm=omega(2, 0);
  float betax=beta(0, 0);
  float betay=beta(1, 0);
  float betaz=beta(2, 0);

  F(0, 0)= 1.0;
  F(0, 1)=-0.5*(pm -dp)*dt;
  F(0, 2)=-0.5*(qm -dq)*dt;
  F(0, 3)=-0.5*(rm -dr)*dt;
  F(0, 4)= 0.5*q1*dt;
  F(0, 5)= 0.5*q2*dt;
  F(0, 6)= 0.5*q3*dt;

  F(1, 0)= 0.5*(pm -dp)*dt;
  F(1, 1)= 1.0;
  F(1, 2)= 0.5*(rm -dr)*dt;
  F(1, 3)=-0.5*(qm -dq)*dt;
  F(1, 4)=-0.5*q0*dt;
  F(1, 5)= 0.5*q3*dt;
  F(1, 6)=-0.5*q2*dt;

  F(2, 0)= 0.5*(qm -dq)*dt;
  F(2, 1)=-0.5*(rm -dr)*dt;
  F(2, 2)= 1.0;
  F(2, 3)= 0.5*(pm -dp)*dt;
  F(2, 4)=-0.5*q3*dt;
  F(2, 5)=-0.5*q0*dt;
  F(2, 6)= 0.5*q1*dt;
  
  F(3, 0)= 0.5*(rm -dr)*dt;
  F(3, 1)= 0.5*(qm -dq)*dt;
  F(3, 2)=-0.5*(pm -dp)*dt;
  F(3, 3)= 1.0;
  F(3, 4)= 0.5*q2*dt;
  F(3, 5)=-0.5*q1*dt;
  F(3, 6)=-0.5*q0*dt;
  
  F(4, 0)= 0.0;
  F(4, 1)= 0.0;
  F(4, 2)= 0.0;
  F(4, 3)= 0.0;
  F(4, 4)= 1.0 -betax*dt;
  F(4, 5)= 0.0;
  F(4, 6)= 0.0;
  
  F(5, 0)= 0.0;
  F(5, 1)= 0.0;
  F(5, 2)= 0.0;
  F(5, 3)= 0.0;
  F(5, 4)= 0.0;
  F(5, 5)= 1.0 -betay*dt;
  F(5, 6)= 0.0;
  
  F(6, 0)= 0.0;
  F(6, 1)= 0.0;
  F(6, 2)= 0.0;
  F(6, 3)= 0.0;
  F(6, 4)= 0.0;
  F(6, 5)= 0.0;
  F(6, 6)= 1.0 -betaz*dt;
  
  return 0;
}

uint8_t H_jacobian(Matrix<float, 6, 7>&H, Matrix<float, 7, 1>x, float g, float mn, float md){
  float q0 = x(0, 0);
  float q1 = x(1, 0);
  float q2 = x(2, 0);
  float q3 = x(3, 0);

  H(0, 0) =-2.0*q2*g;
  H(0, 1) = 2.0*q3*g;
  H(0, 2) =-2.0*q0*g;
  H(0, 3) = 2.0*q1*g;
  H(0, 4) = 0.0;
  H(0, 5) = 0.0;
  H(0, 6) = 0.0;
  
  H(1, 0) = 2.0*q1*g;
  H(1, 1) = 2.0*q0*g;
  H(1, 2) = 2.0*q3*g;
  H(1, 3) = 2.0*q2*g;
  H(1, 4) = 0.0;
  H(1, 5) = 0.0;
  H(1, 6) = 0.0;
  
  H(2, 0) = 2.0*q0*g;
  H(2, 1) =-2.0*q1*g;
  H(2, 2) =-2.0*q2*g;
  H(2, 3) = 2.0*q3*g;
  H(2, 4) = 0.0;
  H(2, 5) = 0.0;
  H(2, 6) = 0.0;
  
  H(3, 0) = 2.0*( q0*mn - q2*md);
  H(3, 1) = 2.0*( q1*mn + q3*md);
  H(3, 2) = 2.0*(-q2*mn - q0*md);
  H(3, 3) = 2.0*(-q3*mn + q1*md);
  H(3, 4) = 0.0;
  H(3, 5) = 0.0;
  H(3, 6) = 0.0;
  
  H(4, 0) = 2.0*(-q3*mn + q1*md);
  H(4, 1) = 2.0*( q2*mn + q0*md);
  H(4, 2) = 2.0*( q1*mn + q3*md);
  H(4, 3) = 2.0*(-q0*mn + q2*md);
  H(4, 4) = 0.0;
  H(4, 5) = 0.0;
  H(4, 6) = 0.0;
  
  H(5, 0) = 2.0*( q2*mn + q0*md);
  H(5, 1) = 2.0*( q3*mn - q1*md);
  H(5, 2) = 2.0*( q0*mn - q2*md);
  H(5, 3) = 2.0*( q1*mn + q3*md);
  H(5, 4) = 0.0;
  H(5, 5) = 0.0;
  H(5, 6) = 0.0;
  
  return 0;
}

uint8_t ekf( Matrix<float, 7, 1> &x,
             Matrix<float, 7, 7> &P,
             Matrix<float, 6, 1> z,
             Matrix<float, 3, 1> omega,
             Matrix<float, 3, 3> Q, 
             Matrix<float, 6, 6> R, 
             Matrix<float, 7, 3> G,
             Matrix<float, 3, 1> beta,
             float dt)
{

  uint8_t ret;
  Matrix<float, 7, 7> F;
  Matrix<float, 6, 7> H;
  Matrix<float, 6, 6> Den;
  Matrix<float, 6, 6> I6=MatrixXf::Identity(6,6);
  Matrix<float, 7, 6> K;
  Matrix<float, 6, 1> zbar;

  
  //Update
  H_jacobian(H, x, GRAV, MN, MD);
  Den = H*P*H.transpose()+R;
  //PartialPivLU< Matrix<float, 6, 6> > dec(Den);
  //Den = dec.solve(I6);
  K = P*H.transpose()*Den.inverse();
  observation_equation(x, zbar, GRAV, MN, MD);
  x = x + K*(z - zbar);
  P = P - K*H*P;
  //std::cout << K << std::endl;
  //std::cout << z-zbar << std::endl;

  //Predict
  state_equation(x, omega, beta, dt);
  F_jacobian(F, x, omega, beta, dt);
  P = F*P*F.transpose() + G*Q*G.transpose();
  return 0;
}

int main(void){
  Matrix<float, 7 ,1> x = MatrixXf::Zero(7,1);
  Matrix<float, 7 ,1> x_sim = MatrixXf::Zero(7,1);
  Matrix<float, 7 ,7> P = MatrixXf::Identity(7,7)*100.0;
  Matrix<float, 6 ,1> z = MatrixXf::Zero(6,1);
  Matrix<float, 6 ,1> z_sim = MatrixXf::Zero(6,1);
  Matrix<float, 3, 1> omega = MatrixXf::Zero(3, 1);
  Matrix<float, 3, 3> Q = MatrixXf::Identity(3, 3)*1.0;
  Matrix<float, 6, 6> R = MatrixXf::Identity(6, 6)*1.0;
  Matrix<float, 7 ,3> G;
  Matrix<float, 3 ,1> beta;
  float t=0.0,dt=0.0025;
  uint64_t s_time=0,e_time=0;
  short i,waittime=15;
  Matrix<float, 4, 1> quat_sim;
  Matrix<float, 3, 1> omega_sim;
  Matrix<float, 3, 1> domega;

  //Variable Initalize
  quat_sim<<1.0, 0.0, 0.0, 0.0;
  x << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  x_sim = x;
  omega << 3.14, 3.14, 3.14;
  omega_sim=omega;
  observation_equation(x, z_sim, GRAV, MN, MD);
  G << 0,0,0, 0,0,0, 0,0,0, 0,0,0, 1,0,0, 0,1,0, 0,0,1;
  beta << 0.0,0.0,0.0;


  //Initilize Console Input&Output
  stdio_init_all();  
  
  //Initilize White Noise
  std::random_device rnd;
  std::mt19937 mt(rnd());  
  std::normal_distribution<> norm(0.0, 1.0);

  //Start up wait
  for (i=0;i<waittime;i++)
  {
    printf("#Please wait %d[s] ! Random number test:%f\n",waittime-i, norm(mt) );
    sleep_ms(1000);
  }
  printf("#Start Kalman Filter\n");
  
  
  //Main Loop 
  while(1)
  {
    printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %llu\n",
              t, 
              x(0,0), x(1,0), x(2,0),x(3,0), x(4,0), x(5,0),x(6,0),
              quat_sim(0,0), quat_sim(1,0), quat_sim(2,0), quat_sim(3,0),
              z_sim(0,0), z_sim(1,0), z_sim(2,0),
              e_time-s_time);  

    //Control
    domega<<x(4,0), x(5,0), x(6,0);
    omega=omega_sim-domega;
    

    //Simulation
    observation_equation(x_sim, z_sim, GRAV, MN, MD);
    z=z_sim;
    rk4(quatdot, t, dt, quat_sim, omega_sim);
    x_sim << quat_sim(0,0), quat_sim(1,0), quat_sim(2,0), quat_sim(3,0), 0,0,0; 
    t=t+dt;

    //Begin Extended Kalman Filter
    s_time=time_us_64();
    ekf(x, P, z, omega, Q, R, G, beta, dt);
    e_time=time_us_64();
    //End   Extended Kalman Filter


    //sleep_ms(100);
  } 
  return 0;
}
