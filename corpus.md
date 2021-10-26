# Reservoirs control problem

## Operational Trade-offs in Reservoir Control

- Reservoir operation decisions require constant revaluation in the face of conflicting objectives, varying hydrologic conditions, and the frequent operational policy changes.

- In systems with multiple objectives the existence of one solution superior with respect to all objectives is very unlikely. This is especially true in reservoirs where conflicts almost always exist between hydropower and flood control, water supply and recreation, and hydropower and water conservation.
- With varying hydrologic conditions, reservoir operation decisions require constant revaluation in the face of frequent operational policy changes.

## Water Reservoir Control with Data Mining

- data mining is the search for relationships and global patterns that exist among parameters, but are hidden among the data.

## Reservoir Optimization in Water Resources: a Review

- Some of the main tasks for water resources management are to supply enough water to the people and at the same time protect them from water related disasters such as flood and landslide.
- Under normal condition, there is a conflict in water management where the storage needs to be ready for incoming flood storage while the demand for water supply still needs to be satisfied.
- Most researchers concentrate on finding the best optimization algorithm for the reservoir problem compared to dam operators who look into operational strategies that are practical to be used.
- To close the gap between theoretical and real world implementation in reservoir optimization, water managers should gain more knowledge on how the reservoir release was operated in the past.
- However the decision on which method to be used depends on the type of problem analysis for each water resources system.
- Despite the increasing number of new optimization algorithms in the literature each year, every method has its own advantages and disadvantages depending on the types of problems involved in water resources especially in reservoir operation.

- The value of variable in the water resource analysis cannot always be specific because of the uncertainty or the fuzziness properties of the variable itself.

## Real-Coded Genetic Algorithm for Rule-Based Flood Control Reservoir Management

- Due to the temporal and spatial variability in rainfall and high mountains and steep channels upstream of all watersheds on Taiwan Island, water reservoirs are the most effective means for mitigating natural disasters such as flooding or drought.
- Operating rules for reservoir operation are intended to guide and manage such systems so that the releases made are in the best interests of the system's objectives consistent with certain inflows and existing storage levels.
- Due to the natural uncertainties of the predicted inflow hydrograph and complexity of reservoir operating rules, reservoir operation highly depends on the experience based knowledge of operator(s), such as during an extreme hydrologic situation like flooding.

## Real-time reservoir flood control operation for cascade reservoirs using a two-stage flood risk analysis method

- Reservoir play an important role in the planning and management of water resources, and the real-time operation of multi-reservoir systems is one of the most vital issues in the field of flood control management.
- Reservoir real-time operations generally involve decision-making for future periods (such as reservoir water level traces or the reservoir release process), and are implemented on the basis of forecasted inflows.
- However, the uncertainty caused by stochastic streamflow leads to a gap between the derived optimal decisions and the actual conditions, resulting in risk events (e.g., underestimation of the magnitude of forecast inflow for reservoir in the system).

## Simulating Reservoir Operation Using a Recurrent Neural Network Algorithm

- Given that the main operating obtain of XLD hydropower is flood control and power generation, this paper focuses on the flood control and power generation benefits of the reservoir operating models.
- However, since the operation time of the XLD reservoir is short (built in 2014) and the inflow of the XLD reservoir is regulated by the upstream reservoir, there are few extreme inflow scenarios in the actual operation of XLD.

- This also results in the model training data set not including extreme inflow events, so the model training fails to take extreme inflow conditions into account.

# Deep Gaussian processes

[Neil D. Lawrence's blog](http://inverseprobability.com)

## [Neil D. Lawrence's Talks](http://inverseprobability.com/talks/notes/gaussian-processes.html)

- In practice, we normally also have uncertainty associated with these functions. Uncertainty in the prediction function arises from
  - scarcity of training data and
  - mismatch between the set of prediction functions we choose and all possible prediction functions.

## Uncertainty Quantification

- Uncertainty quantification (UQ) is the science of quantitative characterization and reduction of uncertainties in both computational and real world applications. It tries to determine how likely certain outcomes are if some aspects of the system are not exactly known.

# [Introduction to Gaussian Processes](http://bridg.land/posts/gaussian-processes-1)

Gaussian processes may not be at the center of current machine learning  hype but are still used at the forefront of research – they were  recently seen automatically tuning the MCTS hyperparameters for AlphaGo  Zero for instance. 

# Gaussian processes for machine learning

## introduction

Indeed, many models that are commonly employed in both machine learning and statistics are in fact special cases of, or restricted kinds of Gaussian processes.

## Regression

### An Example Application

- The inputs were linearly rescaled to have zero mean and unit variance on the training set.
- The outputs were centered so as to have zero mean on the training set.
