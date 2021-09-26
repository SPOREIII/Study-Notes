# Model predictive control

# Nominal model

- [In practice, what is the difference between the “nominal plant model” and the “plant model”?](https://math.stackexchange.com/questions/3368737/in-practice-what-is-the-difference-between-the-nominal-plant-model-and-the-p)

  You design your control for the nominal model, i.e., for the model  where you assume that you know all the parameters, or where you use some averages of what the parameters should be.

  In practice, you use only this model, and no other model is  available. However, the real plant (for sure) differs from your model:  it can have different values of the parameters (parametric  uncertainties) or some unmodelled dynamics, e.g., the senor's inertia.  These uncertainties define how and why the real plant is not the same as the nominal one that you have used for control design, and why your  real signals are not as good as the simulations results. 

  At this moment, you ask: well, can I check in advance how does my  controller deal with all these uncertainties? Yes, to do so you need a  model of the plant that is different from the nominal one, e.g., it  considers possible saturations.  Then you test you nominal controller  (the one that you have designed for the nominal plant) for various plant models with different parameters and so on. It gives you some ideas  about the sensitivity of your controller with respect to uncertainties.  And finally, you apply your controller to the real plant and hope it  will work.

  To summarize, you use the nominal model for control design, you use  plant models to test your design against uncertainties, and you apply  your controller to a real plant.  

