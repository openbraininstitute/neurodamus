"""Module to load configuration from a libsonata config"""

import itertools
import json
import logging
from enum import Enum

import libsonata


class ConnectionTypes(str, Enum):
    Synaptic = "Synaptic"
    GapJunction = "GapJunction"
    NeuroModulation = "NeuroModulation"
    NeuroGlial = "NeuroGlial"
    GlioVascular = "GlioVascular"


class SonataConfig:
    __slots__ = (
        "_circuit_conf",  # libsonata.CircuitConfig
        "_circuits",
        "_sim_conf",  # libsonata.SimulationConfig
        # libsonata returns a set; it should return a list keeping the order of the population names
        "_stable_edge_populations",
        # Currently, the `inputs` of a simulation_config is a json object,
        # which is unordered; however, the order of stimuli matter, so try and
        # recover the order defined in the json file: this assumes that `json.load`
        # keeps it; *which is not guaranteed* see the discussion:
        # https://github.com/openbraininstitute/neurodamus/issues/217#issuecomment-2827930163
        # the SONATA specification should be updated to be a list
        "_stable_inputs_order",
        # libsonata returns a set; it should return a list keeping the order of the population names
        "_stable_node_populations",
    )

    def __init__(self, config_path):
        self._sim_conf = libsonata.SimulationConfig.from_file(config_path)

        with open(config_path, encoding="utf-8") as fd:
            if inputs := json.load(fd).get("inputs", None):
                self._stable_inputs_order = tuple(inputs.keys())
            else:
                self._stable_inputs_order = ()

        with open(self._sim_conf.network, encoding="utf-8") as fd:
            config = json.load(fd)
            self._stable_node_populations = list(
                itertools.chain.from_iterable(e["populations"] for e in config["networks"]["nodes"])
            )
            self._stable_edge_populations = list(
                itertools.chain.from_iterable(e["populations"] for e in config["networks"]["edges"])
            )

        self._circuit_conf = libsonata.CircuitConfig.from_file(self._sim_conf.network)
        self._circuits = self._extract_circuits_info()

    @property
    def beta_features(self):
        return self._sim_conf.beta_features

    @property
    def parsedRun(self):
        item_translation = {
            # Mandatory
            "tstop": "Duration",
            "dt": "Dt",
            "random_seed": "BaseSeed",
            # Optional
            "tstart": "Start",
            "spike_threshold": "SpikeThreshold",
            "integration_method": "SecondOrder",
            "electrodes_file": "LFPWeightsPath",
        }
        parsed_run = self._translate_dict(item_translation, self._sim_conf.run)

        self._adapt_libsonata_fields(parsed_run)

        # "OutputRoot" and "SpikesFile" will be read from self._sim_conf.output
        # once libsonata resolves the manifest info
        parsed_run["OutputRoot"] = self._sim_conf.output.output_dir
        parsed_run["config_node_sets_file"] = self._circuit_conf.node_sets_path
        parsed_run["TargetFile"] = self._circuit_conf.node_sets_path
        parsed_run["SpikesFile"] = self._sim_conf.output.spikes_file
        parsed_run["SpikesSortOrder"] = self._sim_conf.output.spikes_sort_order.name
        parsed_run["Simulator"] = self._sim_conf.target_simulator.name
        parsed_run["TargetFile"] = self._sim_conf.node_sets_file
        parsed_run["CircuitTarget"] = self._sim_conf.node_set
        parsed_run["Celsius"] = self._sim_conf.conditions.celsius
        parsed_run["V_Init"] = self._sim_conf.conditions.v_init
        parsed_run["ExtracellularCalcium"] = self._sim_conf.conditions.extracellular_calcium
        parsed_run["SpikeLocation"] = self._sim_conf.conditions.spike_location.name
        parsed_run["compartment_sets_file"] = self._sim_conf.compartment_sets_file
        return parsed_run

    @property
    def Conditions(self):
        item_translation = {"randomize_gaba_rise_time": "randomize_Gaba_risetime"}
        conditions = {}
        blacklist = (
            "Celsius",
            "VInit",
            "ExtracellularCalcium",
            "ListModificationNames",
            "SpikeLocation",
        )
        for key, value in self._translate_dict(item_translation, self._sim_conf.conditions).items():
            if key in blacklist:
                continue
            if key == "Mechanisms":
                for suffix, dict_var in value.items():
                    for name, val in dict_var.items():
                        conditions[name + "_" + suffix] = val
            else:
                conditions[key] = value

        conditions["randomize_Gaba_risetime"] = str(conditions["randomize_Gaba_risetime"])

        return {"Conditions": conditions}

    def _extract_circuits_info(self) -> dict:
        """Extract the circuits information from confile file with libsonata.CircuitConfig parser,
        return a dictionary of circuit info as:
        {
            pop_name: { "CellLibraryFile": ...,
                        "CircuitTarget": ...,
                        "MorphologyPath": ...,
                        "MorphologyType": ...,
                        "METypePath": ...,
                        "Engine": ...,
                        "nrnPath": ...,
                        "PopulationType": ...
                }
        }
        It will be used to build the internal circuit structure CircuitConfig in configuration.py
        """
        simulation_nodeset_name = self._sim_conf.node_set or ""
        if not simulation_nodeset_name:
            logging.warning("Simulating all populations from all node files...")

        def find_inner_connectivity(node_pop_name):
            """NOTE: Inner connectivity is a special kind of projection, and represents the circuit
            default set of connections. Even though nowadays we can potentially consider
            all connectivity as projections, under certain circuitry, like NGV, order matters and
            therefore we keep inner connectivity to ensure they are created in the same order,
            respecting engine precedence
            For edges to be considered inner connectivity they must be named "default"
            """
            for edge_pop_name in self._stable_edge_populations:
                edge_storage = self._circuit_conf.edge_population(edge_pop_name)
                edge_properties = self._circuit_conf.edge_population_properties(edge_pop_name)

                inner_pop_name = f"{node_pop_name}__{node_pop_name}__chemical"
                if edge_pop_name == inner_pop_name or (
                    edge_storage.source == edge_storage.target == node_pop_name
                    and edge_properties.type == "chemical"
                    and edge_pop_name == "default"
                ):
                    return edge_properties.elements_path + ":" + edge_pop_name
            return False

        def make_circuit(node_pop_name, node_prop):
            morphology_path = node_prop.morphologies_dir
            morphology_type = "h5" if node_prop.type == "astrocyte" else "swc"
            if node_prop.alternate_morphology_formats:
                if "neurolucida-asc" in node_prop.alternate_morphology_formats:
                    morphology_path = node_prop.alternate_morphology_formats["neurolucida-asc"]
                    morphology_type = "asc"
                elif "h5v1" in node_prop.alternate_morphology_formats:
                    morphology_path = node_prop.alternate_morphology_formats["h5v1"]
                    morphology_type = "h5"

            circuit_config = {
                "CellLibraryFile": node_prop.elements_path,
                "CircuitTarget": node_pop_name + ":" + simulation_nodeset_name,
                "Engine": "NGV" if node_prop.type == "astrocyte" else "METype",
                "METypePath": node_prop.biophysical_neuron_models_dir,
                "MorphologyPath": morphology_path,
                "MorphologyType": morphology_type,
                "PopulationType": node_prop.type,
                "nrnPath": find_inner_connectivity(node_pop_name),
            }

            logging.debug("Circuit config for node pop '%s': %s", node_pop_name, circuit_config)
            return circuit_config

        return {
            pop_name: make_circuit(pop_name, node_prop)
            for pop_name in self._stable_node_populations
            if (node_prop := self._circuit_conf.node_population_properties(pop_name)).type
            != "vasculature"
        }

    @property
    def Circuit(self):
        return self._circuits

    @property
    def parsedProjections(self):
        projection_type_convert = {
            "chemical": ConnectionTypes.Synaptic,
            "electrical": ConnectionTypes.GapJunction,
            "synapse_astrocyte": ConnectionTypes.NeuroGlial,
            "endfoot": ConnectionTypes.GlioVascular,
            "neuromodulatory": ConnectionTypes.NeuroModulation,
        }
        projections = {}

        for edge_pop_name in self._stable_edge_populations:
            edge_pop = self._circuit_conf.edge_population(edge_pop_name)
            edge_properties = self._circuit_conf.edge_population_properties(edge_pop_name)

            pop_type = edge_properties.type
            if pop_type not in projection_type_convert:
                logging.warning("Unhandled synapse type: %s", pop_type)
                continue

            # skip inner connectivity populations
            edge_pop_path = edge_properties.elements_path + ":" + edge_pop_name
            if any(edge_pop_path == c["nrnPath"] for c in self._circuits.values()):
                continue

            pop_type = projection_type_convert.get(pop_type)

            projection = {
                "Path": edge_pop_path,
                "Source": edge_pop.source + ":",
                "Destination": edge_pop.target + ":",
                "Type": pop_type,
            }

            # Reverse projection direction for Astrocyte projection: from neurons to astrocytes
            if pop_type == ConnectionTypes.NeuroGlial:
                projection["Source"], projection["Destination"] = (
                    projection["Destination"],
                    projection["Source"],
                )
            elif pop_type == ConnectionTypes.GlioVascular:
                vasculature_popnames = [
                    name
                    for name in self._stable_node_populations
                    if self._circuit_conf.node_population_properties(name).type == "vasculature"
                ]
                if vasculature_popnames:
                    assert len(vasculature_popnames) == 1
                    projection["VasculaturePath"] = self._circuit_conf.node_population_properties(
                        vasculature_popnames[0]
                    ).elements_path

            projections[f"{edge_pop_name}__{edge_pop.source}-{edge_pop.target}"] = projection

        return projections

    @property
    def parsedConnects(self):
        item_translation = {
            "target": "Destination",
            "modoverride": "ModOverride",
            "synapse_delay_override": "SynDelayOverride",
            "neuromodulation_dtc": "NeuromodDtc",
            "neuromodulation_strength": "NeuromodStrength",
        }
        connects = {
            libsonata_conn.name: self._translate_dict(item_translation, libsonata_conn)
            for libsonata_conn in self._sim_conf.connection_overrides()
        }
        return connects

    @property
    def parsedStimuli(self) -> list:
        """Read the inputs information parsed by libsonata,
        and convert them to the internal parameters used by StimulusManager
        """
        item_translation = {
            "module": "Pattern",
            "input_type": "Mode",
            "random_seed": "Seed",
            "series_resistance": "RS",
            "node_set": "Target",
            "sd_percent": "SDPercent",
            "relative_skew": "RelativeSkew",
        }
        input_type_translation = {
            "spikes": "Current",
            "current_clamp": "Current",
            "voltage_clamp": "Voltage",
            "extracellular_stimulation": "Extracellular",
            "conductance": "Conductance",
        }
        module_translation = {"seclamp": "SEClamp", "subthreshold": "SubThreshold"}

        stimuli = []
        # TODO: loop over self._sim_conf.inputs() list after updating SONATA SPEC and libsonata API,
        # The order of stimulus injection could lead to minor difference on the results
        # so need to preserve it as in the config file
        for name in self._stable_inputs_order:
            stimulus = self._translate_dict(item_translation, self._sim_conf.input(name))
            self._adapt_libsonata_fields(stimulus)
            stimulus["Pattern"] = module_translation.get(
                stimulus["Pattern"], snake_to_camel(stimulus["Pattern"])
            )
            stimulus["Mode"] = input_type_translation.get(stimulus["Mode"], stimulus["Mode"])
            stimulus["Name"] = name
            stimuli.append(stimulus)

        return stimuli

    @property
    def parsedReports(self):
        item_translation = {
            "type": "Type",
            "cells": "Target",
            "sections": "Sections",
            "scaling": "Scaling",
            "compartments": "Compartments",
            "variable_name": "ReportOn",
            "unit": "Unit",
            "dt": "Dt",
            "start_time": "StartTime",
            "end_time": "EndTime",
            "file_name": "FileName",
            "enabled": "Enabled",
        }
        reports = {}
        for name in self._sim_conf.list_report_names:
            rep = self._translate_dict(item_translation, self._sim_conf.report(name))
            # Adapt enums and variable names read from libsonata
            self._adapt_libsonata_fields(rep)
            # Format is SONATA with sonata_config
            rep["Format"] = "SONATA"
            reports[name] = rep
            rep["Scaling"] = snake_to_camel(rep["Scaling"])

        return reports

    @property
    def parsedModifications(self):
        item_translation = {"node_set": "Target"}
        result = {}
        for modification in self._sim_conf.conditions.modifications():
            setting = self._translate_dict(item_translation, modification)
            self._adapt_libsonata_fields(setting)
            result[modification.name] = setting

        return result

    @staticmethod
    def _adapt_libsonata_fields(rep):
        for key in rep:
            # Convert enums to its string representation
            if key in {
                "Type",
                "Sections",
                "Scaling",
                "Compartments",
                "Mode",
                "Pattern",
                "SpikeLocation",
            } and not isinstance(rep[key], str):
                rep[key] = rep[key].name
            # Convert comma separated variable names to space separated
            if key == "ReportOn":
                rep[key] = rep[key].replace(",", " ")
            # Get the int value of the enum
            elif key == "SecondOrder":
                rep[key] = int(rep[key])

    @staticmethod
    def _translate_dict(item_translation, libsonata_obj) -> dict:
        """Translate SONATA/libsonata key names (snake_case) to Neurodamus internal paramters"""
        attrs = [
            x
            for x in dir(libsonata_obj)
            if not x.startswith("__") and not callable(getattr(libsonata_obj, x))
        ]
        result = {}
        for att in attrs:
            key = item_translation.get(att, snake_to_camel(att))
            parsed_value = getattr(libsonata_obj, att)
            if parsed_value is not None:
                result[key] = parsed_value
        return result


def snake_to_camel(word):
    return "".join(x.capitalize() or "_" for x in word.split("_"))
