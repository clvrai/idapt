## Configure Domain Randomization Range

- To add a new set of Domain Randomization parameters edit `training/utils/dr_config.json`
- To chose a set of Domain Randomization parameters use argument `--dr_params_set=[PARAMS_SET_NAME]`

### dr_config reference

- **tex_mod** <br>
  Includes all texture randomization parameters. 
  - **active** <br>
    Enables and disables the texture randomization
  - **geoms** <br>
    List of geom names. If included each geom in the list will be set with a unique randomized texture.
  - **modes** <br>
    List of texture of randomization modes to be used. Choose from "checker", "gradient", "rgb", "noise"
- **color_mod**<br>
  Includes all color randomization parameters
  - **active** <br>
    Enables and disables the color randomization. If no other parameters is specified under color_mod, the module will set each geom in the environment to a unique randomized color.
  - **partial_body_change** (*optional*) <br>
    2d list of geom_ids. If included each geoms in each sublist will be set to a randomized color each episode
  - **full_body_change** (*optional*) <br>
    If included and set to true all geoms in the environment will be set to a unified random color
  - **geoms_change** (*optional*) <br>
    List of geom names. If included only the geoms in the list will have randomized colors and each geom's color will be unique.
- **light_mod** <br>
  Includes all light randomization parameters
  - **active** <br>
    Enables and disables the light randomization.
  - **pos_range** <br>
    2d-list of randomization range of light position shift in xyz coordinate
  - **spec_range, diffuse_range, ambient_range** <br>
    Randomization range of the specular, diffuse, and ambient component of the li
  - **head_light** <br>
    Enables and disables the randomization on headlight.
- **camera_mod** <br>
  Includes all viewpoint randomization parameters
  - **active** <br>
    Enables and disables the viewpoint randomization.
  - **use_default** <br>
    Specifies if the environment is using the default camera
  - **fovy_range** <br>
    Randomization range of fovy.
  - **veiwer_cam_param_targets** <br>
     Required only if default camera is used. Includes all the randomization ranges for all attributes in viewer.
  - **cam_name** <br>
    Required only if default camera is not used. Specifies the name of camera being used.
  - **pos_change_range** <br>
    2d-list of randomization range of light viewpoint shift in xyz coordinate.
  - **ang_jitter_range** <br>
    2d-list of randomization range of camera angle jitter in Euler angles.
- **dynamics_mod** <br>
  Includes all dynamics randomization parameters
  - **active** <br>
    Enables and disables the dynamics randomization.
  - **mass_targets** <br> (*optional*)
    Include a dictionary of all bodies to need to be randomized and their mass randomization parameters
    - **range** <br>
      Randomization range of body mass.
    - **inerta_reset** <br>
      If set to true body inertia will be set to match the randomized body mass
  - **friction_targets** (*optional*) <br> 
    Include a dictionary of all geoms to be randomized and their friction randomization parameters
    - **range** <br>
      Randomization range of geom friction.
  - **action_mod** (*optional*) <br> 
    - **bias** <br>
      Randomization range of action bias.
    - **theta** <br>
      Randomization range of action rotation.
  - **armature_multiplier** (*optional*) <br>
    Randomization range of armature proportional to the current armature value.
    