#include <stdio.h>
#include <stdlib.h>
#include "BVH.h"
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp> // after 
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

void convert_quant(float xx, float yy, float zz,
	               float &qw, float &qx, float &qy,float &qz)
{
	glm::mat4 q0s = glm::eulerAngleZ(glm::radians(xx));
	glm::mat4 q1s = glm::eulerAngleY(glm::radians(yy));
	glm::mat4 q2s = glm::eulerAngleX(glm::radians(zz));
	glm::quat q(q0s * (q1s * q2s));
	printf("%.5lf, %.5lf, %.5lf, %.5lf\n", q.w, q.x, q.y, q.z);

	qw = q.w;
	qx = q.x;
	qy = q.y;
	qz = q.z;
}
int main()
{
	BVH bvhloader;
	bvhloader.Load("TestAA.bvh");
	Frame frame;
	bvhloader.QueryOneFrame(0,  frame);

	int i, j;
	for (i = 0; i < frame.joint_rotations.size(); i++)
	{
		for (j = 0; j < 3; j++)
		{
			//printf("%8.3lf ,", frame.joint_rotations[i].euler_rotation[j]);
		}
		float   qw,  qx, qy, qz;
		convert_quant(frame.joint_rotations[i].euler_rotation[0],
			          frame.joint_rotations[i].euler_rotation[1],
			          frame.joint_rotations[i].euler_rotation[2],
			          qw, qx, qy, qz);
		printf("\n");
	}
	/**/

	if (1)
	{
		/*float w, x, y, z;
		glm::mat4 q0s = glm::eulerAngleZ(glm::radians(170.537549));
		glm::mat4 q1s = glm::eulerAngleY(glm::radians(-1.051628));
		glm::mat4 q2s = glm::eulerAngleX(glm::radians(-175.45039));

		glm::quat q(q0s * (q1s * q2s));
		printf("%.5lf, %.5lf, %.5lf, %.5lf\n", q.w, q.x, q.y, q.z);
		printf("hello world3!\n");
		glm::mat4 myMatrix(glm::toMat4(q));
		glm::extractEulerAngleZYX(myMatrix, x, y, z);
		printf("%.3f, %.3f, %.3f\n", x *180.0 / 3.1415, y *180.0 / 3.1415, z *180.0 / 3.1415);*/
	}

	printf("hello world!\n");
	return 1;
}