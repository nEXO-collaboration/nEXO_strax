import strax
export, __all__ = strax.exporter()
import numpy as np
import pint

units = pint.UnitRegistry()
Q_ = units.Quantity

'''
The plan:
for each hit, create channels. Implicit binning

new data_kind: channel WF.
depends on hits with hit_channels
assemble hit_channels into channel WFs
implement global clock 
'''
# @export
# @strax.takes_config(
#     # strax.Option('local_channel_map', default='/home/brodsky3/nexo/nexo-offline/data/localChannelsMap_6mm.txt',help="local channel map file location"),
#     # strax.Option('tile_map', default='/home/brodsky3/nexo/nexo-offline/data/tilesMap_6mm.txt',help="local channel map file location"),
#     strax.Option('PCD_spacing_xy', default=.1,help="PCD spacing in mm"),
#     strax.Option('hit_nPCDs', default=50,help="number of PCDs"),
# )
# class hit_channels(strax.Plugin):
#     depends_on = 'nest_hits'
#     data_kind='nest_hits'
#     provides = 'hit_channels'
#     def infer_dtype(self):
#         return [(('Hit PCD expectation','hit_PCD_expectation'),np.int32,self.config['hit_nPCDs']),
#                 (('Hit PCD quanta', 'hit_PCD_quanta'), np.int32, self.config['hit_nPCDs']),
#                 ]
#     save_when = strax.SaveWhen.TARGET #only save if it's the final target
#
#     def setup(self):
#         pass
#
#     def compute(self, nest_hits):
#         for hit in nest_hits:
#
#
#
#         return result

@export
@strax.takes_config(
            strax.Option('anode_z', track=True, default=str(400*units.mm), help="Anode z position"),
            strax.Option('drift_speed', track=True, default=str(0.171*units.cm/units.microsecond), help="drift speed"),
            strax.Option('diffusion_coeffT', track=True, default=str(1e2*units.cm**2/units.second), help="diffusion coefficient (transverse)"),
            strax.Option('diffusion_coeffL', track=True, default=str(1e1*units.cm**2/units.second), help="diffusion coefficient (longitudinal)"),
            strax.Option('diffusion_coeffL', track=True, default=str(1e1*units.cm**2/units.second), help="diffusion coefficient (longitudinal)"),
            strax.Option('cathode_z', track=True, default=str(-800*units.mm), help="Cathode z position"),
        )

class Thermalelectrons(strax.OverlapWindowPlugin):
    depends_on = 'nest_hits'
    data_kind = 'thermalelectrons'
    provides = 'thermalelectrons'
    dtype = [
        ('x',np.float,'original x position [mm]'),
        ('y', np.float, 'original y position [mm]'),
        ('z', np.float, 'original z position [mm]'),
        ('time', np.int64, 'original time'),
        ('endtime', np.int64, 'time of collection after drift'),
        ('drifttime', np.float, 'drift time [ns]'),
        ('x_drift', np.float, 'drift x position [mm]'),
        ('y_drift', np.float, 'drift y position [mm]'),
        ('z_drift', np.float, 'drift z position [mm]'),

    ]
    save_when = strax.SaveWhen.TARGET

    def setup(self):
        for key,value in self.config.items():
            try:
                self.config[key] = units(value)
            except pint.UndefinedUnitError:
                pass
    def compute(self,nest_hits,start,end):
        nest_hits_trimmed = nest_hits[nest_hits['time']>self.sent_until] #throw away input that already made output
        result = np.zeros(nest_hits_trimmed['n_electrons'].sum(),dtype=self.dtype)
        if len(result):
            for field in ('x','y','z','time'):
                result[field]=np.repeat(nest_hits_trimmed[field],nest_hits_trimmed['n_electrons'])

            drift_time_original = (self.config['anode_z'] - result['z']*units.mm)/self.config['drift_speed']
            sigmaDiffusion = (np.sqrt(2. * self.config['diffusion_coeffT'] * drift_time_original)).m_as(units.mm)
            result['x_drift'] = np.random.normal(result['x'], sigmaDiffusion)
            result['y_drift'] = np.random.normal(result['y'], sigmaDiffusion)

            sigmaDiffusionz = np.sqrt(2. * self.config['diffusion_coeffT'] * drift_time_original).m_as(units.mm);
            result['z_drift'] = np.random.normal(result['z'], sigmaDiffusionz)
            result['drifttime'] = ((self.config['anode_z'] - result['z_drift']*units.mm)/self.config['drift_speed']).m_as(units.ns) # drift time after diffusion
            result['endtime'] = np.int64(np.ceil(result['time']+ result['drifttime']))
            end = result['endtime'].max()

        chunk_result = self.chunk(start=start,end = end, data=result, data_type=self.provides)
        return chunk_result
    def get_window_size(self):
        return ((self.config['anode_z']-self.config['cathode_z'])/self.config['drift_speed']).m_as(units.ns)


@export
class Test_consumer(strax.Plugin):
    depends_on = ['thermalelectrons']
    provides = 'test_consumer2'
    dtype = strax.time_fields
    def compute(self,chunk_i,thermalelectrons):
        print(f'test2: {thermalelectrons["time"]},  {thermalelectrons["endtime"]},  {chunk_i}')
        return np.zeros(0,self.dtype)