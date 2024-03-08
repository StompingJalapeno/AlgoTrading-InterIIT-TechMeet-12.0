from agents.stablebaselines3_models import DRLAgent as DRLAgent_sb3
from meta.data_processor import DataProcessor


def test(start_date, end_date, ticker_list, data_source, time_interval, technical_indicator_list, drl_lib, env, model_name, if_vix=True, **kwargs):
    DP = DataProcessor(data_source, start_date, end_date, time_interval, **kwargs)
    price_array, tech_array, turbulence_array = DP.run(ticker_list,
                                                        technical_indicator_list,
                                                        if_vix, cache=True)
    data_config = {'price_array': price_array,
                   'tech_array': tech_array,
                   'turbulence_array': turbulence_array} 
    
    #build environment using processed data
    env_instance = env(config=data_config)
    env_instance.set_flag(True)                                                    
    net_dimension = kwargs.get("net_dimension", 2**7)
    cwd = kwargs.get('current_working_dir')
    cwd = cwd+'/'+model_name+'.zip'
    

    if drl_lib == "stable_baselines3":
        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=cwd
        )
        env_instance.save_df_test()
        env_instance.save_df_test_logs()
        return episode_total_assets

    else:
        raise ValueError("DRL library input is NOT supported. Please check.")