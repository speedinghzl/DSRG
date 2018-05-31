#include <Eigen/Core>
#include "densecrf.h"
#include "densecrf_wrapper.h"

DenseCRFWrapper::DenseCRFWrapper(int W, int H, int nlabels)
: W(W), H(H), m_nlabels(nlabels) {
	m_crf = new DenseCRF2D(W, H, nlabels);
}

DenseCRFWrapper::~DenseCRFWrapper() {
	delete m_crf;
}

int DenseCRFWrapper::npixels() { return W * H; }
int DenseCRFWrapper::nlabels() { return m_nlabels; }


void DenseCRFWrapper::add_pairwise_energy(float w1,
			 float theta_alpha_1, float theta_alpha_2,
			 float theta_betta_1, float theta_betta_2, float theta_betta_3,
			 float w2,
			 float theta_gamma_1, float theta_gamma_2,
			 unsigned char* im) {

	m_crf->addPairwiseGaussian(theta_gamma_1, theta_gamma_2, new PottsCompatibility(w2));
	m_crf->addPairwiseBilateral(theta_alpha_1, theta_alpha_2,
				    theta_betta_1, theta_betta_2, theta_betta_3,
				    im,
				    new PottsCompatibility(w1));
}

void DenseCRFWrapper::set_unary_energy(float* unary_costs_ptr) {
	m_crf->setUnaryEnergy(
		Eigen::Map<const Eigen::MatrixXf>(
			unary_costs_ptr, m_nlabels, W * H)
	);
}

void DenseCRFWrapper::map(int n_iters, int* labels) {
	VectorXs labels_vec = m_crf->map(n_iters);
	for (int i = 0; i < (W * H); ++i)
		labels[i] = labels_vec(i);
}

void DenseCRFWrapper::inference(int n_iters, float* probs_out) {
  MatrixXf probs = m_crf->inference(n_iters);
	for (int i = 0; i < npixels(); ++i)
    for (int j = 0; j < nlabels(); ++j)
      probs_out[i * nlabels() + j] = probs(j, i);
}
