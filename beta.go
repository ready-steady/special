package sfunc

import (
	"math"
)

// IncBeta computes the incomplete beta function.
//
// The code is based on a C implementation by John Burkardt.
// http://people.sc.fsu.edu/~jburkardt/c_src/asa109/asa109.html
//
// The original algorithm was published in Applied Statistics and described
// below.
//
// Algorithm AS 63
// http://www.jstor.org/stable/2346797
//
// The function uses the method discussed by Soper (1921). If p is not less
// than (p + q)x and the integral part of q + (1 - x)(p + q) is a positive
// integer, say s, reductions are made up to s times “by parts” using the
// recurrence relation
//
//                     Γ(p+q)
//     I(x, p, q) = ----------- x^p (1-x)^(q-1) + I(x, p+1, q-1)
//                  Γ(p+1) Γ(q)
//
// and then reductions are continued by “raising p” with the recurrence
// relation
//
//                          Γ(p+q)
//     I(x, p+s, q-s) = --------------- x^(p+s) (1-x)^(q-s) + I(x, p+s+1, q-s)
//                      Γ(p+s+1) Γ(q-s)
//
// If s is not a positive integer, reductions are made only by “raising p.”
// The process of reduction is terminated when the relative contribution to the
// integral is not greater than the value of ACU. If p is less than (p + q)x,
// I(1-x, q, p) is first calculated by the above procedure and then I(x, p, q)
// is obtained from the relation
//
//     I(x, p, q) = 1 - I(1-x, p, q).
//
// Soper (1921) demonstrated that the expansion of I(x, p, q) by “parts” and
// “raising p” method as described above converges more rapidly than any other
// series expansions.
func IncBeta(x, p, q, logB float64) float64 {
	const (
		acu = 0.1e-14
	)

	if x <= 0 {
		return 0
	}
	if 1 <= x {
		return 1
	}

	psq := p + q
	pbase, qbase := x, 1-x

	flip := false
	if p < psq*x {
		p, q, pbase, qbase = q, p, qbase, pbase
		flip = true
	}

	term, ai := 1.0, 1.0
	temp := q - ai

	rx := pbase / qbase
	ns := int(q + qbase*psq)
	if ns == 0 {
		rx = pbase
	}

	α := 1.0

	for {
		term = term * temp * rx / (p + ai)

		α += term

		temp = math.Abs(term)
		if temp <= acu && temp <= acu*α {
			break
		}

		ai++
		ns--

		if 0 < ns {
			temp = q - ai
		} else if ns == 0 {
			temp = q - ai
			rx = pbase
		} else {
			temp = psq
			psq++
		}
	}

	// Remark AS R19 and Algorithm AS 109
	// http://www.jstor.org/stable/2346887
	α = α * math.Exp(p*math.Log(pbase)+(q-1)*math.Log(qbase)-logB) / p

	if flip {
		return 1 - α
	} else {
		return α
	}
}

// InvIncBeta computes the inverse of the incomplete beta function.
//
// The code is based on a C implementation by John Burkardt.
// http://people.sc.fsu.edu/~jburkardt/c_src/asa109/asa109.html
//
// The original algorithm was published in Applied Statistics and described
// below.
//
// Algorithm AS 64
// http://www.jstor.org/stable/2346798
//
// An approximation x₀ to x if found from (cf. Scheffé and Tukey, 1944)
//
//     (1 + x₀)/(1 - x₀) = (4*p + 2*q - 2)/χ²(α)
//
// where χ²(α) is the upper α point of the χ² distribution with 2*q degrees
// of freedom and is obtained from Wilson and Hilferty’s approximation (cf.
// Wilson and Hilferty, 1931)
//
//     χ²(α) = 2*q*(1 - 1/(9*q) + y(α) * sqrt(1/(9*q)))**3,
//
// y(α) being Hastings’ approximation (cf. Hastings, 1955) for the upper α
// point of the standard normal distribution. If χ²(α) < 0, then
//
//     x₀ = 1 - ((1 - α)*q*B(p, q))**(1/q).
//
// Again if (4*p + 2*q - 2)/χ²(α) does not exceed 1, x₀ is obtained from
//
//     x₀ = (α*p*B(p, q))**(1/p).
//
// The final solution is obtained by the Newton–Raphson method from the
// relation
//
//     x[i] = x[i-1] - f(x[i-1])/f'(x[i-1])
//
// where
//
//     f(x) = I(x, p, q) - α.
func InvIncBeta(α, p, q, logB float64) float64 {
	const (
		// Remark AS R83
		// http://www.jstor.org/stable/2347779
		sae = -30
	)

	if α <= 0 {
		return 0
	}
	if 1 <= α {
		return 1
	}

	flip := false
	if 0.5 < α {
		α = 1 - α
		p, q = q, p
		flip = true
	}

	x := math.Sqrt(-math.Log(α * α))
	y := x - (2.30753+0.27061*x)/(1+(0.99229+0.04481*x)*x)

	if 1 < p && 1 < q {
		// Remark AS R19 and Algorithm AS 109
		// http://www.jstor.org/stable/2346887
		//
		// For p and q > 1, the approximation given by Carter (1947), which
		// improves the Fisher–Cochran formula, is generally better. For other
		// values of p and q en empirical investigation has shown that the
		// approximation given in AS 64 is adequate.
		r := (y*y - 3) / 6
		s := 1 / (2*p - 1)
		t := 1 / (2*q - 1)
		h := 2 / (s + t)
		w := y*math.Sqrt(h+r)/h - (t-s)*(r+5/6-2/(3*h))
		x = p / (p + q*math.Exp(2*w))
	} else {
		t := 1 / (9 * q)
		t = 2 * q * math.Pow(1-t+y*math.Sqrt(t), 3)
		if t <= 0 {
			x = 1 - math.Exp((math.Log((1-α)*q)+logB)/q)
		} else {
			t = 2 * (2*p + q - 1) / t
			if t <= 1 {
				x = math.Exp((math.Log(α*p) + logB) / p)
			} else {
				x = 1 - 2/(t+1)
			}
		}
	}

	if x < 0.0001 {
		x = 0.0001
	} else if 0.9999 < x {
		x = 0.9999
	}

	// Remark AS R83
	// http://www.jstor.org/stable/2347779
	fpu := math.Pow10(sae)
	acu := fpu
	if exp := int(-5/p/p - 1/math.Pow(α, 0.2) - 13); exp > sae {
		acu = math.Pow10(exp)
	}

	tx, yprev, sq, prev := 0.0, 0.0, 1.0, 1.0

outer:
	for {
		// Remark AS R19 and Algorithm AS 109
		// http://www.jstor.org/stable/2346887
		y = IncBeta(x, p, q, logB)
		y = (y - α) * math.Exp(logB+(1-p)*math.Log(x)+(1-q)*math.Log(1-x))

		// Remark AS R83
		// http://www.jstor.org/stable/2347779
		if y*yprev <= 0 {
			prev = math.Max(sq, fpu)
		}

		// Remark AS R19 and Algorithm AS 109
		// http://www.jstor.org/stable/2346887
		for g := 1.0; ; {
			for {
				adj := g * y
				sq = adj * adj

				if sq < prev {
					tx = x - adj

					if 0 <= tx && tx <= 1 {
						break
					}
				}
				g /= 3
			}

			if prev <= acu || y*y <= acu {
				x = tx
				break outer
			}

			if tx != 0 && tx != 1 {
				break
			}

			g /= 3
		}

		if tx == x {
			break
		}

		x = tx
		yprev = y
	}

	if flip {
		return 1 - x
	} else {
		return x
	}
}

// LogBeta computes the logarithm of the beta function.
func LogBeta(x, y float64) float64 {
	z, _ := math.Lgamma(x + y)
	x, _ = math.Lgamma(x)
	y, _ = math.Lgamma(y)

	return x + y - z
}
