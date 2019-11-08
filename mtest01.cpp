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
	//printf("%8.5lf, %8.5lf, %8.5lf, %8.5lf\n", q.w, q.x, q.y, q.z);

	qw = q.w;
	qx = q.x;
	qy = q.y;
	qz = q.z;
}

 
void quset(float*pt, vector<float> vec)
{
	pt[0] = vec[0]; pt[1] = vec[1];
	pt[2] = vec[2]; pt[3] = vec[3];
}

vector<float> query_quant(Frame&frame, int joint_no )
{
	;
	float   qw, qx, qy, qz;
	int i = joint_no;
	convert_quant(frame.joint_rotations[i].euler_rotation[0],
		frame.joint_rotations[i].euler_rotation[1],
		frame.joint_rotations[i].euler_rotation[2],
		qw, qx, qy, qz);
	vector<float> ret_vec;
	ret_vec.push_back(qw);
	ret_vec.push_back(qx);
	ret_vec.push_back(qy);
	ret_vec.push_back(qz);
	return ret_vec;
}

float query_angle(Frame&frame, int joint_no)
{
	return frame.joint_rotations[joint_no].euler_rotation[0]*3.1415926/180.0;
}

void transMimicAction(Frame&frame, vector<float> &skl_action)
{
	float scale_factor = 0.1;
	skl_action.resize(44);
	skl_action[0] = 1 / 60.0;
	skl_action[1] = frame.rootpos[0] * scale_factor;
	skl_action[2] = frame.rootpos[1] * scale_factor;
	skl_action[3] = frame.rootpos[2] * scale_factor;
	quset(&skl_action[4],  query_quant(frame, 0));//  # hip rotation4D                            //1                         
	quset(&skl_action[8],  query_quant(frame, 1));//  # chest rotation4D                         //2           
	quset(&skl_action[12], query_quant(frame, 2));//  # neck rotation4D                         //3
	quset(&skl_action[16], query_quant(frame, 9));//  # right hip rotation4D                    //4
	skl_action[20] = query_angle(frame, 10); //        //5
	quset(&skl_action[21], query_quant(frame, 11));//  # right ankle rotation4D                 //6
	quset(&skl_action[25], query_quant(frame, 3));//  # right shoulder rotation4D               //7
	skl_action[29] = query_angle(frame, 4); //         //8
	quset(&skl_action[30], query_quant(frame, 13));//  # left hip rotation4D                    //9
	skl_action[34] = query_angle(frame, 14); //        //10
	quset(&skl_action[35], query_quant(frame, 15));//  # left ankle rotation4D                  //11
	quset(&skl_action[39], query_quant(frame, 6));//  # left shoulder rotation4D                //12
	skl_action[43] = query_angle(frame, 7); //         //13
}

int main()
{
	BVH bvhloader;
	bvhloader.Load("TestAA.bvh");
	Frame frame;
	//bvhloader.QueryOneFrame(0,  frame);

	int i, j;
	//"Loop": "wrap",
	FILE* file = fopen("dumpActions.json","wt+");
    fprintf(file, "{ \"Loop\": \"wrap\", \"Frames\": [");
	for (i = 0; i < 500; i++)
	{
		fprintf(file, "[");

		bvhloader.QueryOneFrame(i, frame);
		vector<float> skl_action;
		transMimicAction(frame, skl_action);

		for (j = 0; j < 44; j++)
		{
			if(j!=43)
			   fprintf(file, "%8.4f,", skl_action[j]);
			else
			   fprintf(file, "%8.4f ", skl_action[j]);
		}

		if (i != 499)
			fprintf(file, "],");
		else
			fprintf(file, "]");
		fprintf(file, "\n");
	}

	fprintf(file, "]}"); 
	fclose(file);
	/*for (i = 0; i < frame.joint_rotations.size(); i++)
	{
               printf("##%02i=== ",i);
		for (j = 0; j < 3; j++)
		{
			printf("%8.3lf ,", frame.joint_rotations[i].euler_rotation[j]);
		}
		printf("\n");
	}
        
        printf("================================================\n");

	for (i = 0; i < frame.joint_rotations.size(); i++)
	{
               printf("##%02i=== ",i);
		for (j = 0; j < 3; j++)
		{
			//printf("%8.3lf ,", frame.joint_rotations[i].euler_rotation[j]);
		}
		float   qw,  qx, qy, qz;
		convert_quant(frame.joint_rotations[i].euler_rotation[0],
			          frame.joint_rotations[i].euler_rotation[1],
			          frame.joint_rotations[i].euler_rotation[2],
			          qw, qx, qy, qz);
		//printf("\n");
	}
	*/

	 

	printf("hello world!\n");
	return 1;
}
