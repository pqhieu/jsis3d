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
#include <float.h>
#include "higherorder.h"

HigherOrderPotential::~HigherOrderPotential() {
}
HigherOrderPotential::HigherOrderPotential( const VectorXs & cliques, float alpha ) {
    alpha_ = alpha;
    S_ = 0;
    int N = cliques.rows();
    for (int i = 0; i < N; ++i)
        indices_[cliques[i]].push_back(i);
    S_ = indices_.size();
}
void HigherOrderPotential::apply( MatrixXf & out, const MatrixXf & Q, const VectorXi & mask ) {
    int M = Q.rows();
    int N = Q.cols();
    out.resize( M, N );
    out.fill( 0.0 );
    MatrixXf h_norm( M, S_ );
    h_norm.fill( 0.0 );

    double p;
    int i = 0;
    for ( auto& s : indices_ ) {
        for ( int k = 0; k < M; ++k ) {
            p = 0.0;
            for ( auto& v : s.second ) {
                // if (mask[v]) continue;
                p += Q(k, v) + FLT_EPSILON;
            }
            h_norm(k, i) = p;
        }
        i++;
    }

    i = 0;
    for ( auto& s : indices_ ) {
        float sum = h_norm.col(i).sum();
        for ( auto& v : s.second ) {
            // if (mask[v]) continue;
            for ( int k = 0; k < M; ++k ) {
                p = (h_norm(k, i) - Q(k, v) - FLT_EPSILON) / (sum - Q(k, v) - FLT_EPSILON);
                out(k, v) = -alpha_ * log(fmaxf(p, FLT_EPSILON));
            }
        }
        i++;
    }
}
