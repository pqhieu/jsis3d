/*
    Copyright (c) 2017-2019, Quang-Hieu Pham
    All rights reserved.

    THIS SOFTWARE IS PROVIDED BY Quang-Hieu Pham ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Quang-Hieu Pham BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <map>
#include <vector>
#include <Eigen/Core>
#include "objective.h"
using namespace Eigen;

class HigherOrderPotential {
protected:
    // Number of segments
    int S_;
    std::map< int, std::vector<int> > indices_;
    float alpha_;
	HigherOrderPotential( const HigherOrderPotential &o ){}
public:
	virtual ~HigherOrderPotential();
	HigherOrderPotential(const VectorXs& cliques, float weight);
	void apply(MatrixXf & out, const MatrixXf & Q, const VectorXi & mask);
};
