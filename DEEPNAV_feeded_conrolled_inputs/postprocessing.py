#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:48:47 2020

@author: Ahmed Majuid
"""
import os, csv, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def flight_pdf_plots(file_name, ground_truth, predictions):
    """
    Arguments
        file_name: string name of the required output pdf (including relative path)
        ground_truth: np array of shape (n_time_steps, n_states)
        predictions: same as ground_truth but for NN predictions

    Output
        creates pdf file, each page is a state plotted with time, both truth and
        predictions are plotted
    """

    # timesteps vector
    time_length = ground_truth.shape[0]
    time_steps = np.linspace(0, time_length-1, time_length)

    # convert time from steps to minutes (5 examples are logged per second)
    dt = 0.2
    time_steps = time_steps * dt / 60

    with PdfPages(file_name) as pdf:

        signal_names = ["north velocity (m/s)", "east velocity (m/s)", "down velocity (m/s)", 
                        "north position (m)", "east position (m)", "down position (m)"]

        # loop on signals of this one flight
        for signal_number, signal_name in enumerate(signal_names):
            plt.figure()
            true_signal = ground_truth[:, signal_number]
            predicted_signal = predictions[:, signal_number]

            # plot title is the mean absolute error between the truth and predictions
            signal_MAE = np.mean(np.absolute(true_signal - predicted_signal))

            plt.plot(time_steps, true_signal) 
            plt.plot(time_steps, predicted_signal, linestyle=(0, (1, 1)))
            plt.grid(True)
            plt.ylabel(signal_name)
            plt.xlabel("time (minutes)")
            plt.title("MAE=" + '{:.3f}'.format(signal_MAE))
            plt.legend(("EKF with GPS", "Network without GPS"))   
            pdf.savefig()
            plt.close()


def evaluate_all_flights(model, train_flights_dict, val_flights_dict, trial_folder, n_extreme_flights=10):
    
    """
    Arguments

        model: trained tf model to make the predictions

        train_flights_dict: a dictionary whose key is flight name and value is a tuple of (features,labels)
        val_flights_dict: same but for validation flights
        trial_folder: string name of the trial folder "DeepNav_results/trial_###"
        n_extreme_flights: integer number of flights to be separated as best or worst, for example, if 
                           n_extreme_flights=5 then best (or worst) folder will contain best 5 flights

    return
        flights_summary: a dictionary of two elements (training & validation), the value is a 2D list
                        whose colums are (flight_duration, max_pos_error, max_vel_error)
    Outputs
        - creates one pdf file containing plots of both prediction and ground 
          truth of attitude, velocity and postion for each flight, with these pdfs, the 
          following folders are populated
            # training
            #   |_ differenced
            #   |_ reconstructed
            #      |_best - worst - other
            # validation
            #   |_ differenced
            #   |_ reconstructed
            #      |_best - worst - other
    """

    # loop on sets, one iteration for training and another for validation
    flights_summary = {}
    set_names = ["training","validation"]
    for flights_dict, set_name in zip([train_flights_dict, val_flights_dict], set_names):
        
        # sort flights by name (shorter flight first)
        flights_list = sorted(flights_dict.items())
        total_flights = len(flights_list) - 1

        # dictionary of (flight_name : max_pos_error) pairs, used to extract the best & worst flights
        flights_errors = {}

        # array of shape (time_steps, 3), colums are (flight_duration, max_pos_error, max_vel_error)
        set_summary = []

        for flight_number, one_flight_data in enumerate(flights_list):
            
            ##to speedup experimenting
            # if flight_number > 5:
            #     break

            flight_name = one_flight_data[0]
            print("flight " + str(flight_number) + "/" + str(total_flights) + " : " + flight_name)
            
            features = one_flight_data[1][0]
            ground_truth_diff = one_flight_data[1][1]
            predictions_diff = model.predict(features)

            # Reconstruct the original signals from differenced signals
            ground_truth_reconstructed = np.cumsum(ground_truth_diff, axis = 0)
            predictions_reconstructed = np.cumsum(predictions_diff, axis = 0)

            # reconstructed output csv file name 
            output_csv_file_nn = os.path.join(trial_folder, set_name, "reconstructed", \
                                              "nn_output_csv", flight_name + "_nn.csv")

            # differenced output csv file name 
            output_csv_file_nn_diff = os.path.join(trial_folder, set_name, "differenced", \
                                              "nn_output_csv", flight_name + "_nn.csv")
            
            # save the reconstructed predictions (ground truth already saved by create_dataset.py)
            np.savetxt(output_csv_file_nn, predictions_reconstructed, delimiter=",")

            # save the differenced predictions
            np.savetxt(output_csv_file_nn_diff, predictions_diff, delimiter=",")

            # maximum errors between prediction and ground truth
            max_velocity_error = np.max(np.linalg.norm(ground_truth_reconstructed[:,0:3] \
                                                      -predictions_reconstructed[:,0:3], axis=1))
            max_position_error = np.max(np.linalg.norm(ground_truth_reconstructed[:,3:6] \
                                                      -predictions_reconstructed[:,3:6], axis=1))

            # add error to the output file name
            pdf_name = flight_name + "_MPE_" f'{max_position_error:.2f}' + \
                                     "_MVE_" f'{max_velocity_error:.2f}' + ".pdf"

            # create a pdf for this flight differenced signals
            pdf_name_diff = os.path.join(trial_folder, set_name, "differenced", pdf_name)
            flight_pdf_plots(pdf_name_diff, ground_truth_diff, predictions_diff)

            # create a pdf for this flight reconstructed signals
            pdf_name_recon = os.path.join(trial_folder, set_name, "reconstructed", "other", pdf_name)
            flight_pdf_plots(pdf_name_recon, ground_truth_reconstructed, predictions_reconstructed)

            flights_errors[pdf_name] = max_position_error

            flight_duration = ground_truth_reconstructed.shape[0] * 0.2 / 60
            set_summary.append([int(flight_name[0:4]), flight_duration, max_position_error, max_velocity_error])
        
        flights_summary[set_name] = set_summary
        
        # sort the flights from by position error (min error first)
        sorted_flights = sorted(flights_errors.items(), key=lambda x: x[1])
        
        # move the pdfs of best & worst flights of this set to their respective folders
        old_name_base = os.path.join(trial_folder, set_name, "reconstructed", "other")
        best_name_base = os.path.join(trial_folder, set_name, "reconstructed", "best")
        worst_name_base = os.path.join(trial_folder, set_name, "reconstructed", "worst")

        for i in range(n_extreme_flights):
            pdf_name = sorted_flights[i][0]
            old_name = os.path.join(old_name_base, pdf_name)
            new_name = os.path.join(best_name_base, pdf_name)
            os.rename(old_name, new_name)
        
        for i in range(-n_extreme_flights, 0):
            pdf_name = sorted_flights[i][0]
            old_name = os.path.join(old_name_base, pdf_name)
            new_name = os.path.join(worst_name_base, pdf_name)
            os.rename(old_name, new_name)
    
    return flights_summary