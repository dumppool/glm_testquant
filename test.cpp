#include <stdio.h>
#include <stdlib.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp> // after 
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

int main()
{
 /*glm::mat4 myMatrix = glm::translate(glm::mat4(), glm::vec3(10.0f, 0.0f, 0.0f));
 glm::vec4 myVector(10.0f, 10.0f, 10.0f, 0.0f);
 glm::vec4 transformedVector = myMatrix * myVector;*/
 //glm::mat4 myMatrix = glm::eulerAngleZYX(170.537549, -1.051628, -175.45039);
 //glm::mat4 
 //glm::quat q();//(myMatrix);
 //printf("%.2, %.2, %.2, %.2\n", q.w, q.x, q.y, q.z);
 
 
 glm::mat4 myMatrix = glm::eulerAngleXYZ(170.537549, -1.051628, -175.45039);
 glm::quat q(myMatrix);
 printf("%.3f, %.3f, %.3f, %.3f\n",  q.x, q.y, q.z, q.w);
 printf("hello world!\n");
 
 myMatrix = glm::eulerAngleXYZ(-175.45039, -1.051628, 170.537549);
 glm::quat q1(myMatrix);
 printf("%.3f, %.3f, %.3f, %.3f\n",  q1.x, q1.y, q1.z, q1.w);
 printf("hello world!\n");
 //---------------------------------------------------------------------------
 printf("============================================\n");
 if(1)
 {
 glm::mat4 myMatrix = glm::eulerAngleXYZ( glm::radians(170.537549), glm::radians(-1.051628), glm::radians(-175.45039));
 glm::quat q(myMatrix);
 printf("%.3f, %.3f, %.3f, %.3f\n",  q.x, q.y, q.z, q.w);
 printf("hello world!\n");
 
 myMatrix = glm::eulerAngleXYZ(glm::radians(-175.45039), glm::radians(-1.051628), glm::radians(170.537549));
 glm::quat q1(myMatrix);
 printf("%.3f, %.3f, %.3f, %.3f\n",  q1.x, q1.y, q1.z, q1.w);
 printf("hello world!\n");
 }
 printf("============================================\n");
 if(1)
 {
  float w,x,y,z;
  glm::mat4 q0s =  glm::eulerAngleZ(glm::radians(170.537549));
  glm::mat4 q1s =  glm::eulerAngleY(glm::radians(-1.051628));
  glm::mat4 q2s =  glm::eulerAngleX(glm::radians(-175.45039));
  
  glm::quat q(q0s * (q1s * q2s));
  printf("%.5lf, %.5lf, %.5lf, %.5lf\n",  q.w, q.x, q.y, q.z );
  printf("hello world3!\n");
  glm::mat4 myMatrix(glm::toMat4(q));
  glm::extractEulerAngleZYX(myMatrix, x,y,z);
  printf("%.3f, %.3f, %.3f\n", x *180.0/3.1415, y *180.0/3.1415, z *180.0/3.1415);
 }
 /*glm::mat4 myMatrix1 = glm::eulerAngleX( 170.537549);
 glm::mat4 myMatrix2 = glm::eulerAngleY( -1.051628 );
 glm::mat4 myMatrix3 = glm::eulerAngleZ( -175.45039);*/
 
 return 1;
}