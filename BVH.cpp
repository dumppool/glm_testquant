#include <fstream>
#include <string.h>
#include <math.h>
#include "BVH.h"


// コントラクタ
BVH::BVH()
{
	motion = NULL;
	Clear();
}

BVH::BVH(const char * bvh_file_name)
{
	motion = NULL;
	Clear();
	Load(bvh_file_name);
}

BVH::~BVH()
{
	Clear();
}


void  BVH::Clear()
{
	int  i;
	for (i = 0; i < channels.size(); i++) delete  channels[i];
	for (i = 0; i < joints.size(); i++)  delete  joints[i];
	if (motion != NULL) delete  motion;
	is_load_success = false;
	file_name = ""; motion_name = "";
	num_channel = 0;
	channels.clear();
	joints.clear();
	joint_index.clear();
	num_frame = 0;
	interval = 0.0;
	motion = NULL;
}

 
/* 
void  BVH::GenAnimationFigure(int frame_no, float scale, Frame&oneframe)
{
	vector<Joint> Jontvec;
	QueryJoint(joints[0], motion + frame_no * num_channel, Jontvec);
}

// 指定フレームの姿勢を描画
void  BVH::RenderFigure(int frame_no, float scale)
{
	// BVH骨格・姿勢を指定して描画

}


// 
void  BVH::QueryJoint( Joint * joint, const double * data, vector<Joint>&Jontvec)
{
	double scale(0);
	// 親関節からの回転を適用（ルート関節の場合はワールド座標からの回転）
	int  i, j;
	printf("=================Idx:%i===NAME:%s===================\n",joint->index, joint->name.c_str());
	for (i = 0; i < joint->channels.size(); i++)
	{
	
		Channel *  channel = joint->channels[i];
		joint->euler_rotation[i] = data[channel->index];
		if (channel->type == X_ROTATION)
		{
			glRotatef(data[channel->index], 1.0f, 0.0f, 0.0f);
			printf("ROTX: %.2lf ,", data[channel->index]);
		}
		else if (channel->type == Y_ROTATION)
		{
			glRotatef(data[channel->index], 0.0f, 1.0f, 0.0f);
			printf("ROTY: %.2lf ,", data[channel->index]);
		}
		else if (channel->type == Z_ROTATION)
		{
			glRotatef(data[channel->index], 0.0f, 0.0f, 1.0f);
			printf("ROTZ: %.2lf ,", data[channel->index]);
		} 
	}
	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	// リンクを描画
	// 関節座標系の原点から末端点へのリンクを描画
	if (joint->children.size() == 0)
	{
		//RenderBone(0.0f, 0.0f, 0.0f, joint->site[0] * scale, joint->site[1] * scale, joint->site[2] * scale);
	}
	// 関節座標系の原点から次の関節への接続位置へのリンクを描画
	if (joint->children.size() == 1)
	{
		Joint *  child = joint->children[0];
		//RenderBone(0.0f, 0.0f, 0.0f, child->offset[0] * scale, child->offset[1] * scale, child->offset[2] * scale);
	}
	// 全関節への接続位置への中心点から各関節への接続位置へ円柱を描画
	if (joint->children.size() > 1)
	{
		// 原点と全関節への接続位置への中心点を計算
		float  center[3] = { 0.0f, 0.0f, 0.0f };
		for (i = 0; i < joint->children.size(); i++)
		{
			Joint *  child = joint->children[i];
			center[0] += child->offset[0];
			center[1] += child->offset[1];
			center[2] += child->offset[2];
		}
		center[0] /= joint->children.size() + 1;
		center[1] /= joint->children.size() + 1;
		center[2] /= joint->children.size() + 1;

		// 原点から中心点へのリンクを描画
		//RenderBone(0.0f, 0.0f, 0.0f, center[0] * scale, center[1] * scale, center[2] * scale);

		// 中心点から次の関節への接続位置へのリンクを描画
		for (i = 0; i < joint->children.size(); i++)
		{
			Joint *  child = joint->children[i];
			//RenderBone(center[0] * scale, center[1] * scale, center[2] * scale,
			//	child->offset[0] * scale, child->offset[1] * scale, child->offset[2] * scale);
		}
	}

	// 子関節に対して再帰呼び出し
	for (i = 0; i < joint->children.size(); i++)
	{
		//RenderFigure(joint->children[i], data, scale);
		QueryJoint(joint, data, Jontvec);
	}

	//glPopMatrix();
}

 */


// End of BVH.cpp