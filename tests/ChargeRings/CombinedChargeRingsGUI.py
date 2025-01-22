#!/usr/bin/python

import sys
import os
import json
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from TipMultipole import (
    makeCircle, makeRotMats, compute_site_energies,
    compute_site_tunelling, makePosXY, compute_V_mirror
)

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Combined Charge Rings GUI v2")
        self.main_widget = QtWidgets.QWidget(self)
        
        # Initialize calculation results storage
        self.tip_potential_data = None
        self.qdot_system_data = None
        
        # Parameter specifications (same as before)
        self.param_specs = {
            # Tip Parameters
            'Q_tip':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (-2.0, 2.0),  'value': 0.6, 'step': 0.1},
            'VBias':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.0, 2.0),   'value': 1.0, 'step': 0.1},
            'Rtip':          {'group': 'Tip Parameters',    'widget': 'double', 'range': (0.5, 5.0),   'value': 1.0, 'step': 0.1},
            'z_tip':         {'group': 'Tip Parameters',    'widget': 'double', 'range': (1.0, 20.0),  'value': 6.0, 'step': 0.5},
            
            # System Parameters
            'cCouling':      {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 1.0),   'value': 0.02, 'step': 0.01, 'decimals': 3},
            'temperature':   {'group': 'System Parameters', 'widget': 'double', 'range': (0.1, 100.0), 'value': 10.0, 'step': 1.0},
            'onSiteCoulomb': {'group': 'System Parameters', 'widget': 'double', 'range': (0.0, 10.0),  'value': 3.0,  'step': 0.1},
            
            # Mirror Parameters
            'zV0':           {'group': 'Mirror Parameters', 'widget': 'double', 'range': (-5.0, 0.0),  'value': -2.5, 'step': 0.1},
            'zQd':           {'group': 'Mirror Parameters', 'widget': 'double', 'range': (-5.0, 5.0),  'value': 0.0,  'step': 0.1},
            
            # Ring Geometry
            'nsite':         {'group': 'Ring Geometry',     'widget': 'int',    'range': (1, 10),       'value': 3},
            'radius':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (1.0, 20.0),   'value': 5.0, 'step': 0.5},
            'phiRot':        {'group': 'Ring Geometry',     'widget': 'double', 'range': (-10.0, 10.0), 'value': -1.0,'step': 0.1},
            
            # Site Properties
            'Esite':         {'group': 'Site Properties',   'widget': 'double', 'range': (-10.0, 10.0), 'value': -1.0,'step': 0.1},
            'Q0':            {'group': 'Site Properties',   'widget': 'double', 'range': (-10.0, 10.0), 'value': 1.0, 'step': 0.1},
            'Qzz':           {'group': 'Site Properties',   'widget': 'double', 'range': (-20.0, 20.0), 'value': 0.0, 'step': 0.5},
            
            # Visualization
            'L':             {'group': 'Visualization',     'widget': 'double', 'range': (5.0, 50.0),  'value': 20.0, 'step': 1.0},
            'npix':          {'group': 'Visualization',     'widget': 'int',    'range': (50, 500),    'value': 200,  'step': 50},
            'decay':         {'group': 'Visualization',     'widget': 'double', 'range': (0.1, 2.0),   'value': 0.7,  'step': 0.1,   'decimals': 2},
            'dQ':            {'group': 'Visualization',     'widget': 'double', 'range': (0.001, 0.1), 'value': 0.02, 'step': 0.001, 'decimals': 3},
        }
        
        # Dictionary to store widget references
        self.param_widgets = {}
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        # --- Main Layout
        l00 = QtWidgets.QHBoxLayout(self.main_widget)
        
        # --- Matplotlib Canvas
        self.fig = Figure(figsize=(15, 10))
        self.canvas = FigureCanvas(self.fig)
        l00.addWidget(self.canvas)
        
        # --- Control Panel
        l0 = QtWidgets.QVBoxLayout(); l00.addLayout(l0)
        
        # Create widgets for each parameter group
        current_group = None
        current_layout = None
        
        for param_name, spec in self.param_specs.items():
            # Create new group if needed
            if spec['group'] != current_group:
                current_group = spec['group']
                gb = QtWidgets.QGroupBox(current_group); l0.addWidget(gb)
                current_layout = QtWidgets.QVBoxLayout(gb)
            
            # Create widget layout
            hb = QtWidgets.QHBoxLayout(); current_layout.addLayout(hb)
            hb.addWidget(QtWidgets.QLabel(f"{param_name}:"))
            
            # Create appropriate widget type
            if spec['widget'] == 'double':
                widget = QtWidgets.QDoubleSpinBox()
                widget.setRange(*spec['range'])
                widget.setValue(spec['value'])
                widget.setSingleStep(spec['step'])
                if 'decimals' in spec:
                    widget.setDecimals(spec['decimals'])
            elif spec['widget'] == 'int':
                widget = QtWidgets.QSpinBox()
                widget.setRange(*spec['range'])
                widget.setValue(spec['value'])
                if 'step' in spec:
                    widget.setSingleStep(spec['step'])
            
            widget.valueChanged.connect(self.run_simulation)
            hb.addWidget(widget)
            self.param_widgets[param_name] = widget
        
        # Controls
        hb = QtWidgets.QHBoxLayout(); l0.addLayout(hb)
        
        # Auto-update checkbox
        cb = QtWidgets.QCheckBox("Auto-update"); cb.setChecked(True); hb.addWidget(cb); self.cbAutoUpdate=cb
        
        # Run Button
        btn = QtWidgets.QPushButton("Run Simulation"); btn.clicked.connect(self.run_simulation); hb.addWidget(btn)
        
        # Save/Load buttons
        hb = QtWidgets.QHBoxLayout(); l0.addLayout(hb)
        btnSave = QtWidgets.QPushButton("Save Parameters"); btnSave.clicked.connect(self.save_parameters); hb.addWidget(btnSave)
        btnLoad = QtWidgets.QPushButton("Load Parameters"); btnLoad.clicked.connect(self.load_parameters); hb.addWidget(btnLoad)
        
        # Set the central widget and initialize
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.init_simulation()

        self.run_simulation()

    
    def get_param_values(self):
        """Get current values of all parameters"""
        return {name: widget.value() for name, widget in self.param_widgets.items()}
    
    def set_param_values(self, values):
        """Set values for all parameters"""
        for name, value in values.items():
            if name in self.param_widgets:
                self.param_widgets[name].setValue(value)
    
    def save_parameters(self):
        """Save parameters to JSON file"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Parameters", "", "JSON files (*.json)")
        if filename:
            if not filename.endswith('.json'):
                filename += '.json'
            with open(filename, 'w') as f:
                json.dump(self.get_param_values(), f, indent=4)
    
    def load_parameters(self):
        """Load parameters from JSON file"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Parameters", "", "JSON files (*.json)")
        if filename:
            with open(filename, 'r') as f:
                values = json.load(f)
                self.set_param_values(values)
                self.run_simulation()

    def init_simulation(self):
        """Initialize simulation with Python backend"""
        params = self.get_param_values()
        nsite = params['nsite']
        R = params['radius']
        
        # Setup sites on circle using Python implementation
        self.spos, phis = makeCircle(n=nsite, R=R)
        self.spos[:,2] = params['z_tip']
        
        # Setup multipoles and site energies
        self.Esite = np.full(nsite, params['Esite'])
        self.rots = makeRotMats(phis + params['phiRot'])
        
        # Initialize global parameters
        self.temperature = params['temperature']
        self.onSiteCoulomb = params['onSiteCoulomb']
    
    def calculateTipPotential(self, params):
        """Calculate tip potential data for X-Z projections"""
        # X-Z grid
        ps_xz, Xs, Zs = makePosXY(n=params['npix'], L=params['L'], axs=(0,2,1))
        
        # Tip position
        zT = params['z_tip'] + params['Rtip']
        tip_pos = np.array([0.0, 0.0, zT])
        
        # Calculate potentials
        self.tip_potential_data = {
            'Vtip':   compute_V_mirror(tip_pos, ps_xz,  VBias=params['VBias'],  Rtip=params['Rtip'], zV0=params['zV0']).reshape(params['npix'], params['npix']),
            'Esites': compute_site_energies( ps_xz, np.array([[0.0,0.0,params['zQd']]]), params['VBias'], params['Rtip'], zV0=params['zV0']).reshape(params['npix'], params['npix']),
            'ps_xz':  ps_xz,
            'extent': [-params['L'], params['L'], -params['L'], params['L']]
        }
        
        # Calculate 1D potential along x at z=0
        ps_1d = np.zeros((params['npix'], 3))
        ps_1d[:,0] = np.linspace(-params['L'], params['L'], params['npix'])  # x coordinates
        ps_1d[:,2] = 0.0  # z=0 for all points
        self.tip_potential_data['V1d'] = compute_V_mirror(tip_pos, ps_1d, VBias=params['VBias'], Rtip=params['Rtip'], zV0=params['zV0'])

    def calculateQdotSystem(self, params):
        """Calculate quantum dot system data for X-Y projections"""
        # X-Y grid
        ps_xy, Xs, Ys = makePosXY(n=params['npix'], L=params['L'], p0=(0,0,params['z_tip']))
        
        # Compute site energies and tunneling rates
        Es = compute_site_energies( ps_xy, self.spos,VBias=params['Q_tip'], Rtip=1.0, zV0=params['z_tip'], E0s=self.Esite )
        
        T = compute_site_tunelling(ps_xy, self.spos, beta=params['decay'], Amp=1.0).reshape(params['npix'], params['npix'], -1)
        
        # Calculate charge distribution
        Q = self.solve_occupancies(Es, T.reshape(-1, T.shape[-1]))
        Q = Q.reshape(params['npix'], params['npix'], -1)
        
        # Compute STM and dI/dQ
        I = self.getSTM_map(Q, T, params['dQ'])
        dIdQ = (I - np.mean(I)) / params['dQ']
        
        self.qdot_system_data = { 'total_charge': np.sum(Q, axis=2), 'STM': I, 'dIdQ': dIdQ, 'ps_xy': ps_xy, 'extent': [-params['L'], params['L'], -params['L'], params['L']] }
    
    def plotTipPotential(self):
        """Plot X-Z projections using precomputed data"""
        data = self.tip_potential_data
        params = self.get_param_values()
        
        # 1D Potential
        self.ax1.clear()
        x_coords = np.linspace(-data['extent'][1], data['extent'][1], params['npix'])
        self.ax1.plot(x_coords, data['V1d'])
        self.ax1.set_title("1D Potential (z=0)")
        self.ax1.set_xlabel("x [Å]")
        self.ax1.set_ylabel("V [V]")
        self.ax1.grid()
        
        # Tip Potential
        self.ax2.clear()
        self.ax2.imshow(data['Vtip'], extent=data['extent'], cmap='bwr', origin='lower', vmin=-params['VBias'], vmax=params['VBias'])
        # Add circles for tip radius surface
        zT = params['z_tip'] + params['Rtip']
        circ1, _ = makeCircle(16, R=params['Rtip'], axs=(0,2,1), p0=(0.0,0.0,zT))
        circ2, _ = makeCircle(16, R=params['Rtip'], axs=(0,2,1), p0=(0.0,0.0,2*params['zV0']-zT))
        self.ax2.plot(circ1[:,0], circ1[:,2], ':k')
        self.ax2.plot(circ2[:,0], circ2[:,2], ':k')
        self.ax2.set_title("Tip Potential")
        self.ax2.set_xlabel("x [Å]")
        self.ax2.set_ylabel("z [Å]")
        self.ax2.grid()
        
        # Site Potential
        self.ax3.clear()
        self.ax3.imshow(data['Esites'], extent=data['extent'], cmap='bwr', origin='lower', vmin=-params['VBias'], vmax=params['VBias'])
        self.ax3.axhline(params['zV0'], ls='--', c='k', label='mirror surface')
        self.ax3.axhline(params['zQd'], ls='--', c='g', label='Qdot height')
        self.ax3.legend()
        self.ax3.set_title("Site Potential")
        self.ax3.set_xlabel("x [Å]")
        self.ax3.set_ylabel("z [Å]")
        self.ax3.grid()
    
    def plotQdotSystem(self):
        """Plot X-Y projections using precomputed data"""
        data = self.qdot_system_data
        params = self.get_param_values()
        
        # Energies (using bwr colormap)
        self.ax4.clear()
        Es = compute_site_energies(data['ps_xy'], self.spos, VBias=params['Q_tip'], Rtip=1.0, zV0=params['z_tip'], E0s=self.Esite)
        Es = Es.reshape(params['npix'], params['npix'], -1)
        total_energy = np.sum(Es, axis=2)  # Sum over sites
        im = self.ax4.imshow(total_energy, origin="lower", extent=data['extent'], cmap='bwr')
        self.ax4.plot(self.spos[:,0], self.spos[:,1], '+g')
        self.fig.colorbar(im, ax=self.ax4)
        self.ax4.set_title("Energies")
        self.ax4.set_xlabel("x [Å]")
        self.ax4.set_ylabel("y [Å]")
        
        # Site Charge
        self.ax5.clear()
        im = self.ax5.imshow(data['total_charge'], origin="lower", extent=data['extent'])
        self.ax5.plot(self.spos[:,0], self.spos[:,1], '+g')
        self.fig.colorbar(im, ax=self.ax5)
        self.ax5.set_title("Site Charge")
        self.ax5.set_xlabel("x [Å]")
        self.ax5.set_ylabel("y [Å]")
        
        # STM
        self.ax6.clear()
        im = self.ax6.imshow(data['STM'], origin="lower", extent=data['extent'])
        self.ax6.plot(self.spos[:,0], self.spos[:,1], '+g')
        self.fig.colorbar(im, ax=self.ax6)
        self.ax6.set_title("STM")
        self.ax6.set_xlabel("x [Å]")
        self.ax6.set_ylabel("y [Å]")
    
    def run_simulation(self):
        """Run simulation using Python backend"""
        self.init_simulation()
        params = self.get_param_values()
        
        # Create 2x3 grid of plots
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 3)
        
        # Top row: X-Z projections
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # 1D Potential
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # Tip Potential
        self.ax3 = self.fig.add_subplot(gs[0, 2])  # Site Potential
        
        # Bottom row: X-Y projections
        self.ax4 = self.fig.add_subplot(gs[1, 0])  # Total Charge
        self.ax5 = self.fig.add_subplot(gs[1, 1])  # STM
        self.ax6 = self.fig.add_subplot(gs[1, 2])  # dI/dQ
        
        # Perform calculations
        self.calculateTipPotential(params)
        self.calculateQdotSystem(params)
        
        # Update all plots
        self.plotTipPotential()
        self.plotQdotSystem()
        
        # Adjust layout and draw
        self.fig.tight_layout()
        self.canvas.draw()

    def solve_occupancies(self, Es, T):
        """Calculate site occupancies using Fermi-Dirac statistics"""
        kT = 8.617e-5 * self.temperature  # eV/K
        nsite = Es.shape[1]
        occupancies = 1/(1 + np.exp((Es - 0.5*self.onSiteCoulomb)/kT)) 
        return occupancies.reshape((-1, nsite))

    def getSTM_map(self, Q, T, dQ):
        """Calculate STM image from charge distribution"""
        Q_perturbed = Q * (1 + dQ)
        return np.sum(Q_perturbed * T, axis=-1)  # Sum over last dimension

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())