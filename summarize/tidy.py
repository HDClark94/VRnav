# tidy up some legend names and axis names into better names



def legend_title_experiment(experiment):
    if experiment == 'basic_settings_cued_RLincrem_ana_1s_0':
        return "No Uncertainty"
    elif experiment == 'basic_settings_cued_RLincrem_ana_1s_05':
        return "Low Uncertainty"
    elif experiment == 'basic_settings_cued_RLincrem_ana_1s_2':
        return "High Uncertainty"
    elif experiment == 'basic_settings_non_cued_RLincrem_button_tap':
        return "Button tap"
    elif experiment == 'basic_settings_RLimcrem_analogue':
        return "Joystick"

def yaxis_mean_error_title(error_type):
    if error_type == 'first_stop_error':
        return "Mean First Stop Error (VU)"
    elif error_type == 'absolute_first_stop_error':
        return "Absolute Mean First Stop Error (VU)"

def yaxis_variance_error_title(error_type):
    if error_type == 'first_stop_error':
        return "First Stop Error SD (VU)"
    elif error_type == 'absolute_first_stop_error':
        return "Absolute First Stop Error SD (VU)"

def trial_type_title(trial_type):
    if trial_type == "non_beaconed":
        return "Non Beaconed"
    elif trial_type == "beaconed":
        return "Beaconed"