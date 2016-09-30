#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include "Brain.h"

#define ADDRESS_NDB "/Users/toosyou/ext/Research/neuron_data/linesetLiWarpTransformRelease_updated0816.ndb"
#define ADDRESS_TIPS_BRANCHES "tips_branches"

using namespace std;

int main(int argc, char const *argv[]) {
    Brain sample(ADDRESS_NDB);

    mkdir(ADDRESS_TIPS_BRANCHES, 0755);

    //make tips files
    for(int i=0;i<sample.size();++i){
        string name_neuron = sample[i].name();
        vector< vector<float> > vertices_tips = sample[i].tips_original();

        //output tips files
        char file_name[200]={0};
        sprintf(file_name, "%s/%d.tips", ADDRESS_TIPS_BRANCHES, i);

        fstream out_tips(file_name, fstream::out);
        out_tips << name_neuron <<endl;

        for(int j=0;j<vertices_tips.size();++j){
            for(int k=0;k<3;++k){
                out_tips << vertices_tips[j][k];
                if( k==2 )
                    out_tips << endl;
                else
                    out_tips << " ";
            }
        }
    }

    //make branches files
    for(int i=0;i<sample.size();++i){
        string name_neuron = sample[i].name();
        vector< vector<float> > vertices_branches = sample[i].branches_original();

        //output tips files
        char file_name[200]={0};
        sprintf(file_name, "%s/%d.brc", ADDRESS_TIPS_BRANCHES, i);

        fstream out_tips(file_name, fstream::out);
        out_tips << name_neuron <<endl;

        for(int j=0;j<vertices_branches.size();++j){
            for(int k=0;k<3;++k){
                out_tips << vertices_branches[j][k];
                if( k==2 )
                    out_tips << endl;
                else
                    out_tips << " ";
            }
        }
    }


    return 0;
}
