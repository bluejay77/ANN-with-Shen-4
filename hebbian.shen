\\ AJY 2016-02-24
\\
\\ From George F Luger: ARTIFICIAL INTELLIGENCE, 4th Edition
\\ Pearson--Addison-Wesley, ISBN 0-201-64866-0
\\
\\ This file discusses the Hebbian learning.
\\
\\ The program has been implemented with vectors, and with the non
\\ type secure defstruct.shen library.
\\ It is one of the author's early ANN endeavours, and does not
\\ have significant commercial value; it is here as an example
\\ of an ANN program implemented with vectors.
\\ 
\\ 10.5.3 Supervised Hebbian learning, pp. 450 --
\\
\\ a) Unsupervised Hebbian learning:
\\
\\ D(W) = c * f(X, W) * X
\\
\\ D(W) == the adjustment vector to the weights vector W
\\ c    == a small learning constant, here 0.2
\\ f(X, W) == the output vector of the network nodes, for the input
\\ vector X and the weight vector W
\\ 
\\ b) supervised Hebbian learning:
\\
\\ D(W) = c * DO * X
\\
\\ D(W) == the adjustment vector to the weights vector W
\\ c == ie. 0.2
\\ DO== desired output vector, for training the network
\\ X == the input vector, training data
\\
\\ and the training instances are a set of vector pairs <X, Y>
\\
\\ therefore
\\
\\ D(w(i,k)) == c * d(k) * x(i)
\\
\\ where 
\\
\\ D(w(i,k)) == the adjustment to the ith input weight, to the kth node,
\\ in the output layer
\\ c == ie. 0.2
\\ d(k) == the desired output of the kth output node
\\ x(i) == the ith element of X, the training data
\\
\\ and, as the training instances are a set of vector pairs <X, Y>,
\\ then d(k) == Y(i,k)
\\
\\ In other words:
\\
\\ D(W) = c * Y * X
\\ where
\\ D(W) == weights adjustment array
\\ c == training constant, ie. 0.2
\\ Y == desired output vector, wanted results vector
\\ X == inputs vector, to be associated with vector Y
\\ with Hebbian learning
\\ and the Y * X is the outer vector product Y[1,n] * X[1,m]:
\\
\\ Y * X = (@v (@v y1*x1 y1*x2 y1*x3 ... y1*xm <>)
\\             (@v y2*x1 y2*x2 y2*x3 ... y2*xm <>)
\\             (@v y3*x1 y3*x2 y3*x3 ... y3*xm <>)
\\             ...
\\             (@v yn*x1 yn*x2 yn*x3 ... yn*xm <>)
\\             <>)
\\
\\ Usage:
\\
\\ LOADING INSTRUCTIONS--
\\
\\ First load Ramil Farkshatov's defstruct package:
\\
\\ (load "defstruct.shen")
\\
\\ Secondly, load Ramil Farkshatov's FOR loop macro:
\\
\\ (load "tc_for.shen")
\\
\\ After that, load and run this file:
\\
\\ Load Willi Riha's mathematics library:
\\
\\ (load "maths-lib.shen")
\\
\\ Load the arrays maths aux package:
\\
\\ (load "array.shen")
\\
\\ Load this file:
\\
\\ (load "hebbian.shen")
\\ 
\\ and run the exercise functions, namely:
\\
\\ Run (assign-hebbian) for the initial values of the weights.
\\ Run (train-hebbian 100) to train the Hebbian net.
\\
\\ Run (test-hebbian) and give it some increasing and decreasing
\\ integer sequences
\\
\\ Here, make a Hebbian net of 6 inputs and 3 outputs, and
\\ try to train it to recognize increasing integer sequences.
\\
\\ Which nets can ne trained to recognize which tasks, is a
\\ research question.  It turns out that the bare Hebbian net
\\ can be trained to recognize -- roughly -- an increasing
\\ integer sequance.
\\
\\ AJY 2016-01-09
\\
\\ Author's Note: Therefore, this program is really a scientific
\\ experiment.  Its real value is educational, as a preparation
\\ for the Parts #2, #3 and #4 in this program series,
\\ to minimize functions with the Hopfield networks with Shen.

\\ Training data:

(set x-y (@v (@p (@v 1 2 3 4 5 6 <>) (@v 1 1 1 <>))
             (@p (@v 6 5 4 3 2 1 <>) (@v -1 -1 -1 <>))
             (@p (@v 0 0 0 0 0 0 <>) (@v -1 -1 -1 <>))
             (@p (@v 5 1 2 8 5 3 <>) (@v -1 -1 -1 <>))
             (@p (@v 20 30 40 50 60 70 <>) (@v 1 1 1 <>))
	     (@p (@v -1 2 -55 30 11 0 <>) (@v -1 -1 -1 <>))
	     (@p (@v -10 -9 -8 -7 -6 -5 <>) (@v 1 1 1 <>))
             <>))


\\ Define the Hebbian ANN


(defstruct hebbian
  (nr-inputs number)
  (inputs-vec vector)
  (weights-vec vector)
  (activation-level number)
  (treshold-function (number --> number))
  (neuron-output number))


\\ Calling (transfer-function N) where N is a neuron, is equivalent
\\ with firing the neuron N, ie. computing its output


(define transfer-function
  { hebbian --> number }
  N ->
    (let
      AL (activation-level N)
      _  (hebbian-activation-level-> N AL)
      F  (hebbian-treshold-function N)
      _  (hebbian-neuron-output-> N (F (hebbian-activation-level N)))
      (hebbian-neuron-output N)))


(define activation-level
  { hebbian --> number }
  N -> (activation-level-h N 1 0))


(define activation-level-h
  { hebbian --> number --> number --> number }
  N Counter Sum ->
    Sum where (> Counter (hebbian-nr-inputs N))
  N Counter Sum ->
    (activation-level-h
      N (+ 1 Counter)
        (+ Sum (* (<-vector (hebbian-inputs-vec N) Counter)
                  (<-vector (hebbian-weights-vec N) Counter)))))


\\ The hebbian treshold function -- the signum function (modified)

(define treshold
  { number --> number }
  Activation -> 1 where (>= Activation 0)
  Activation -> -1)



\\ Create the Hebbian neuron net: 3 artificial neurons, and each and
\\ every one has 6 inputs
\\
\\ Now the intention is to train the net to recognize increasing
\\ integer sequences.
\\ We could eg. carry out a majority operation on the outputs of
\\ the three Hebbian neurons in question to decide if we have an
\\ increasing integer sequence in our hands.


(set hebbian-ann (@v 
                   (mk-hebbian
                       6
                       (vector 6)
                       (vector 6)
                       0
                       (function treshold)
                       0)
                   (mk-hebbian
                       6
                       (vector 6)
                       (vector 6)
                       0
                       (function treshold)
                       0)
                   (mk-hebbian
                       6
                       (vector 6)
                       (vector 6)
                       0
                       (function treshold)
                       0)
                   <>))


\\ Assign values for weights: all are set arbitrarily at 0.7
\\

(define assign-hebbian
  { --> (list A) }
  ->
  (do
    (for (Indx 1 1 3)
      (let HN (<-vector (value hebbian-ann) Indx)
           _ (hebbian-weights-vec-> HN
                                  (@v 0.7 0.7 0.7 0.7 0.7 0.7 <>))
	   _ (vector-> (value hebbian-ann) Indx HN)
	   [] \\ Dummy for the (let ...) form
      ) \\ end (let ...)
    ) \\ end (for ...)
  []) \\ end (do ...)
) \\ end function




\\ Now, define the function to train the hebbian-ann net
\\


(define train-hebbian
  { number --> (list A) }
  NrRounds ->
    (do \\ Train net, then return []
      (for (Round 1 1 NrRounds) \\ Train with the material NrRounds times
	   (train-hebbian-ann)
      ) \\ end (for ...)
      [] \\ Return []
    ) \\ end (do ...)
) \\ end function


(define train-hebbian-ann
    { --> (list A) }
    ->
      (let
	  C 0.2 \\ learning constant
	  Items (limit (value x-y)) \\ # of training vector pairs
	  _ (for (Item 1 1 Items) \\ All of training vector pairs
	    (let
	      Y (snd (<-vector (value x-y) Item)) \\ D, desired result
	      X (fst (<-vector (value x-y) Item)) \\ input for D
	      V (outer-product Y X)
	      DW (scalar-mult C V)
	      _ (for (Neuron 1 1 3) \\ thru all 3 Hebbian neurons
                (let
		    N (<-vector (value hebbian-ann) Neuron) \\ Store it
		      \\ First compute update, then assign it
		      _ (let
			  W (hebbian-weights-vec N) \\ Neuron N's weight vec
			  _ (let
			      _ (for (Weight 1 1 6) \\ Now update weights for N
				     (let
					 WT (<-vector W Weight)
					 \\ D(w(i,k)) == c * d(k) * x(i)
					 DWik (* C
						 (<-vector Y Neuron)
						 (<-vector X Weight))
					 WT (+ WT DWik) \\ Update
					 (vector-> W Weight WT)
					 ) \\ end (let ...)
				) \\ end (for (Weight 1 6) ...)
				  \\ Updated weights are in vector W
				  \\ Assign the W into the vector hebbian-ann
				UN (<-vector (value hebbian-ann) Neuron) 
				_ (hebbian-weights-vec-> UN W)
				_ (vector-> (value hebbian-ann) Neuron UN)
				[]) \\ end (let ...)
			  [])  \\ end (let ...)
		    []) \\ end (let ...)
	       ) \\ end (for (Neuron 1 3) ...)
            []) \\ end (let ...)
          ) \\ end (for (Item 1 Items) ...)
        []) \\ end outermost (let ...)
) \\ end (define train-hebbian ...)



\\ ------------------------------------------------------------
\\ Aux vector operations, transferred to the file array.shen

\\ ------------------------------------------------------------
\\ The function to test the trained neuron
\\

(define test-hebbian
  { --> (list A) }
  ->
    (let
      _ (output "~%Give a sequence of six integers: ")
      InL (lineread)
      _ (output "~%Numbers: ~A~%" InL)
      I (list2vec InL)
      _ (output "~%Vector: ~A~%" I)
      N1 (<-vector (value hebbian-ann) 1)
      N2 (<-vector (value hebbian-ann) 2)
      N3 (<-vector (value hebbian-ann) 3)
      _ (hebbian-inputs-vec-> N1 I)
      _ (hebbian-inputs-vec-> N2 I)
      _ (hebbian-inputs-vec-> N3 I)
      O1 (transfer-function N1)
      O2 (transfer-function N2)
      O3 (transfer-function N3)
      _ (output "~%~%Output of ANN: ~A~%" [O1 O2 O3])
      []))


\\ End result:
\\
\\ Run (assign-hebbian) for initial values of weights.
\\ Run (train-hebbian 100) to train the Hebbian net.
\\
\\ Run (test-hebbian) and give it the increasing sequence
\\    1 2 3 4 5 6
\\
\\ The resulting output of the Hebbian ANN:
\\
\\ Output of ANN: [1 1 1]
\\
\\ Run (test-hebbian) and give it the decreasing sequence
\\    -1 -2 -3 -4 -5 -6
\\
\\ The resulting output of the Hebbian ANN:
\\
\\ Output of ANN: [-1 -1 -1]
\\
\\ Success!  We have (roughly) trained the net to recognize
\\ increasing integer sequences.  Roughly.  (This bare, basic Hebbian
\\ net works but it does not work 100%.)

\\ Aux functions
\\
\\ ------------------------------------------------------------
\\


(define list2vec
  { (list number) --> (vector number) }
  L ->
    (let
      Ln (length L)
      VEC (vector Ln)
      (list2vec-h L VEC Ln 1)))


(define list2vec-h
  { (list number) --> (vector number) --> number --> number
     --> (vector number) }
  L VEC Ln Indx ->
    VEC where (> Indx Ln)
  L VEC Ln Indx ->
    (let
      _ (vector-> VEC Indx (hd L))
      (list2vec-h (tl L) VEC Ln (+ 1 Indx))))

