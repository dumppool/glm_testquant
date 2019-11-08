/* Copyright (c) Russell Gillette
* December 2013
*
* Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
* documentation files (the "Software"), to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
* to permit persons to whom the Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
* BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <fstream>
#include <algorithm>
#include "render/DrawMesh.h"
#include "render/DrawUtil.h"

cDrawMesh::cDrawMesh() : mNumElem(0)//, mVbos(0)
{
}

cDrawMesh::~cDrawMesh()
{
}

void cDrawMesh::Init(int num_buffers)
{
	 
}

void cDrawMesh::Draw(GLenum primitive)
{
	Draw(primitive, 0);
}

void cDrawMesh::Draw(GLenum primitive, int start_idx)
{
	 
}

void cDrawMesh::Draw(GLenum primitive, int idx_start, int idx_end)
{
	 
}

void cDrawMesh::AddBuffer(int buff_num)
{
	 
}

/*void cDrawMesh::LoadVBuffer(unsigned int buffer_num, int data_size, GLubyte *data, int data_offset, int num_attr, tAttribInfo *attr_info)
{
	 
}*/

void cDrawMesh::LoadIBuffer(int num_elem, int elem_size, int *data)
{ 
}

// by having a range, we can choose to only update the buffers we have changed
// extent is the number of buffers after the base index to update
void cDrawMesh::SyncGPU(unsigned int base, size_t extent)
{
	 
}

int cDrawMesh::GetNumFaces() const
{
	//const int verts_per_face = 3;
	//int num_faces = mIbo.GetNumElems() / verts_per_face;
	return 1;//num_faces;
}

int cDrawMesh::GetNumVerts() const
{
	/*int num_verts = 0;
	if (mVbos.size() > 0)
	{
		const cVertexBuffer& vbos = mVbos[0];
		int comp = vbos.mAttrInfo->mNumComp;
		int attr_size = vbos.mAttrInfo->mAttribSize;
		int size = vbos.mSize;
		num_verts = size / (comp * attr_size);
	}*/
	return 1;//num_verts;
}