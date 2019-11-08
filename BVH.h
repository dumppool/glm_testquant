#ifndef  _BVH_H_
#define  _BVH_H_
#include <vector>
#include <map>
#include <string.h>
#include <string>
#include <fstream>
#include <string.h>
#define  BUFFER_LENGTH  1024*32
using namespace  std;


struct  Joint;
enum  ChannelEnum
{
	X_ROTATION, Y_ROTATION, Z_ROTATION,
	X_POSITION, Y_POSITION, Z_POSITION
};
struct  Channel
{ 
	Joint *              joint; // 对应关节 
	ChannelEnum          type;  // 旋转の種類 
	int                  index; // 旋转编号
};

struct  Joint
 {
	    double euler_rotation[3];  
		string               name;     // 关节名
		int                  index;    // 关节编号
		Joint *              parent;   // 关节层次（父关节）
		vector< Joint * >    children; // 关节层次（子关节）
		double               offset[3];// 关节耦合位置
		bool                 has_site; // 末端位置情報
		double               site[3];  // 末端位置
		vector< Channel * >  channels; // 旋转轴
 };

int inline cmp_func(const Joint & a,const Joint & b)
{
    if(a.index > b.index)
        return true;
    return false;
}


struct Frame
{
  int frameIdx;
  vector<Joint> joint_rotations;
};

class  BVH
{
public:
	bool                     is_load_success; // 文件是否加载成功
	//  文件的信息   
	string                   file_name;   // 文件名
	string                   motion_name; // 动作名
	// 层次構造的信息  
	int                      num_channel; // 旋转数
	vector< Channel * >      channels;    // 旋转情報 [旋转编号]
	vector< Joint * >        joints;      // 关节情報 [パーツ编号]
	map< string, Joint * >   joint_index; // 关节名字map
	int                      num_frame;   // 动画帧数
	double                   interval;    // 动画帧間の時間間隔
	double *                 motion;      // [动画帧编号][旋转编号]

public:
	//void  GenAnimationFigure(int frame_no, float scale, Frame&oneframe);
	BVH();//  
	BVH(const char * bvh_file_name);
	~BVH();
	 
	void  Clear();
	int inline Load(const char * bvh_file_name);

public:
	// 文件的信息的获取
	const string &  GetFileName() const { return file_name; }
	const string &  GetMotionName() const { return motion_name; }
	// 层次構造的信息的获取
	const int       GetNumJoint() const { return  joints.size(); }
	const Joint *   GetJoint(int no) const { return  joints[no]; }
	const int       GetNumChannel() const { return  channels.size(); }
	const Channel * GetChannel(int no) const { return  channels[no]; }

	const Joint *   GetJoint(const string & j) const {
		map< string, Joint * >::const_iterator  i = joint_index.find(j);
		return  (i != joint_index.end()) ? (*i).second : NULL;
	}
	const Joint *   GetJoint(const char * j) const {
		map< string, Joint * >::const_iterator  i = joint_index.find(j);
		return  (i != joint_index.end()) ? (*i).second : NULL;
	}

	int     GetNumFrame() const { return  num_frame; }
	double  GetInterval() const { return  interval; }
	//double*  GetMotion(int f, int c) const { return  &motion[f*num_channel + c]; }
	double*  GetMotion(int frame_no) const { return  motion + frame_no * num_channel; }
	void inline QueryOneFrame(int Idx, Frame&frame);
public:
 
};

void inline BVH::QueryOneFrame(int Idx, Frame&frame)
{
	double *frame_dat = GetMotion(Idx);
	int i,j;
	frame.frameIdx = Idx;
	frame.joint_rotations.clear();
	for (i = 0; i < joints.size(); i++)
	{
		//printf("%s -", joints[i]->name.c_str() );
		Joint temp = (*joints[i]);
		if (i == 0)
		{
			temp.euler_rotation[0] = frame_dat[joints[i]->channels[3]->index];
			temp.euler_rotation[1] = frame_dat[joints[i]->channels[4]->index];
			temp.euler_rotation[2] = frame_dat[joints[i]->channels[5]->index];
		}
		else
		for (j = 0; j < joints[i]->channels.size(); j++)
		{
			Channel *  joint_channel = joints[i]->channels[j];
			//printf("%i -", joint_channel->type );
			temp.euler_rotation[j] = frame_dat[joint_channel->index];
		}
		//printf("\n");
		frame.joint_rotations.push_back(temp);
	}
}

int inline BVH::Load(const char * bvh_file_name)
{
	ifstream  file;
	char      line[BUFFER_LENGTH];
	char *    token;
	char      separater[] = " :,\t";
	vector< Joint * >   joint_stack;
	Joint *   joint = NULL;
	Joint *   new_joint = NULL;
	bool      is_site = false;
	double    x, y, z;
	int       i, j;
	// 初期化
	Clear();
	// ファイルの情報（ファイル名・动作名）の設定
	file_name = bvh_file_name;
	const char *  mn_first = bvh_file_name;
	const char *  mn_last = bvh_file_name + strlen(bvh_file_name);
	if (strrchr(bvh_file_name, '\\') != NULL)
		mn_first = strrchr(bvh_file_name, '\\') + 1;
	else if (strrchr(bvh_file_name, '/') != NULL)
		mn_first = strrchr(bvh_file_name, '/') + 1;
	if (strrchr(bvh_file_name, '.') != NULL)
		mn_last = strrchr(bvh_file_name, '.');
	if (mn_last < mn_first)
		mn_last = bvh_file_name + strlen(bvh_file_name);
	motion_name.assign(mn_first, mn_last);

	// ファイルのオープン
	file.open(bvh_file_name, ios::in);
	if (file.is_open() == 0)  return false; // ファイルが開けなかったら終了

	// 階層情報の読み込み
	while (!file.eof())
	{
		// ファイルの最後まできてしまったら異常終了
		if (file.eof())
			goto bvh_error;

		// １行読み込み、先頭の単語を取得
		file.getline(line, BUFFER_LENGTH);
		token = strtok(line, separater);

		// 空行の場合は次の行へ
		if (token == NULL)  continue;

		// 関節ブロックの開始
		if (strcmp(token, "{") == 0)
		{
			// 現在の関節をスタックに積む
			joint_stack.push_back(joint);
			joint = new_joint;
			continue;
		}
		// 関節ブロックの終了
		if (strcmp(token, "}") == 0)
		{
			// 現在の関節をスタックから取り出す
			joint = joint_stack.back();
			joint_stack.pop_back();
			is_site = false;
			continue;
		}

		// 関節情報の開始
		if ((strcmp(token, "ROOT") == 0) ||
			(strcmp(token, "JOINT") == 0))
		{
			// 関節データの作成
			new_joint = new Joint();
			new_joint->index = joints.size();
			new_joint->parent = joint;
			new_joint->has_site = false;
			new_joint->offset[0] = 0.0;  new_joint->offset[1] = 0.0;  new_joint->offset[2] = 0.0;
			new_joint->site[0] = 0.0;  new_joint->site[1] = 0.0;  new_joint->site[2] = 0.0;
			joints.push_back(new_joint);
			if (joint)
				joint->children.push_back(new_joint);

			// 関節名の読み込み
			token = strtok(NULL, "");
			while (*token == ' ')  token++;
			new_joint->name = token;

			// インデックスへ追加
			joint_index[new_joint->name] = new_joint;
			continue;
		}

		// 末端情報の開始
		if ((strcmp(token, "End") == 0))
		{
			new_joint = joint;
			is_site = true;
			continue;
		}

		// 関節のオフセット or 末端位置の情報
		if (strcmp(token, "OFFSET") == 0)
		{
			// 座標値を読み込み
			token = strtok(NULL, separater);
			x = token ? atof(token) : 0.0;
			token = strtok(NULL, separater);
			y = token ? atof(token) : 0.0;
			token = strtok(NULL, separater);
			z = token ? atof(token) : 0.0;
			
			// 関節のオフセットに座標値を設定
			if (is_site)
			{
				joint->has_site = true;
				joint->site[0] = x;
				joint->site[1] = y;
				joint->site[2] = z;
			}
			else
				// 末端位置に座標値を設定
			{
				joint->offset[0] = x;
				joint->offset[1] = y;
				joint->offset[2] = z;
			}
			continue;
		}

		if (strcmp(token, "CHANNELS") == 0)
		{
			// チャンネル数を読み込み
			token = strtok(NULL, separater);
			joint->channels.resize(token ? atoi(token) : 0);

			// チャンネル情報を読み込み
			for (i = 0; i < joint->channels.size(); i++)
			{
				// チャンネルの作成
				Channel *  channel = new Channel();
				channel->joint = joint;
				channel->index = channels.size();
				channels.push_back(channel);
				joint->channels[i] = channel;

				// チャンネルの種類の判定
				token = strtok(NULL, separater);
				if (strcmp(token, "Xrotation") == 0)
					channel->type = X_ROTATION;
				else if (strcmp(token, "Yrotation") == 0)
					channel->type = Y_ROTATION;
				else if (strcmp(token, "Zrotation") == 0)
					channel->type = Z_ROTATION;
				else if (strcmp(token, "Xposition") == 0)
					channel->type = X_POSITION;
				else if (strcmp(token, "Yposition") == 0)
					channel->type = Y_POSITION;
				else if (strcmp(token, "Zposition") == 0)
					channel->type = Z_POSITION;
			}
		}

		// Motionデータのセクションへ移る
		if (strcmp(token, "MOTION") == 0)
			break;
	}


	// モーション情報の読み込み
	file.getline(line, BUFFER_LENGTH);
	token = strtok(line, separater);
	if (strcmp(token, "Frames") != 0)
		goto bvh_error;
	token = strtok(NULL, separater);
	if (token == NULL)
		goto bvh_error;
	num_frame = atoi(token);

	file.getline(line, BUFFER_LENGTH);
	token = strtok(line, ":");
	if (strcmp(token, "Frame Time") != 0)
		goto bvh_error;
	token = strtok(NULL, separater);
	if (token == NULL)
		goto bvh_error;
	interval = atof(token);

	num_channel = channels.size();
	motion = new double[num_frame * num_channel];

	// 装填 motion 数据
	for (i = 0; i < num_frame; i++)
	{
		file.getline(line, BUFFER_LENGTH);
		token = strtok(line, separater);
		for (j = 0; j < num_channel; j++)
		{
			if (token == NULL)
				goto bvh_error;
			motion[i*num_channel + j] = atof(token);
			token = strtok(NULL, separater);
		}
	}

	 
	file.close();

	is_load_success = true;

	return is_load_success;

bvh_error:
	return false;
}



#endif // _BVH_H_