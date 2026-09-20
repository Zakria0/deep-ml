def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
	"""
	Calculate posterior probabilities using Bayes' Theorem.
	
	Args:
		priors: Prior probabilities P(H_i) for each hypothesis
		likelihoods: Likelihoods P(E|H_i) for each hypothesis
		
	Returns:
		Posterior probabilities P(H_i|E) for each hypothesis
	"""
	n = len(priors)
	evidence = sum(likelihoods[i] * priors[i] for i in range(n))
	posteriors = []

	for i in range(n):
		numer = likelihoods[i] * priors[i]
		posteriors.append(numer / evidence)
	
	return posteriors
	