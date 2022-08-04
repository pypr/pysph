<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
  <Domain>\
    ${tempotral_collection(files, times, n_particles, particles_info, vectorize_velocity)}\
  </Domain>
</Xdmf>

<%def name="tempotral_collection(files, times, n_particles, particles_info, vectorize_velocity)">
    <Grid Name="temporal" GridType="Collection" CollectionType="Temporal" >
    % for index, (file, time) in enumerate(zip(files,times)):
      <Grid Name="spatial" GridType="Collection" CollectionType="Spatial" >
        <Time Type="Single" Value="   ${time}" />\
        ${spatial_collection(file, index, n_particles, particles_info, vectorize_velocity)}\
      </Grid>
    % endfor
    </Grid>
</%def>

<%def name="spatial_collection(file, index, n_particles, particles_info, vectorize_velocity)">
  % for pname, data in particles_info.items():
        <Grid Name="${pname}" GridType="Uniform">\
          ${topo_and_geom(file, pname, n_particles[pname][index])}\
          ${variables_data(file, pname, n_particles[pname][index], data['output_props'],data['stride'], data['attr_type'], vectorize_velocity)}\
        </Grid>
  % endfor
</%def>

<%def name="topo_and_geom(file, pname, n_particles)">
          <Topology TopologyType="Polyvertex" Dimensions="${n_particles}" NodesPerElement="1"/>
          <Geometry Type="X_Y_Z">
            <DataItem Format="HDF" Dimensions="${n_particles}" NumberType="Float">
              ${file}:/particles/${pname}/arrays/x
            </DataItem>
            <DataItem Format="HDF" Dimensions="${n_particles}" NumberType="Float">
              ${file}:/particles/${pname}/arrays/y
            </DataItem>
            <DataItem Format="HDF" Dimensions="${n_particles}" NumberType="Float">
              ${file}:/particles/${pname}/arrays/z
            </DataItem>
          </Geometry>
</%def>

<%def name="variables_data(file, pname, n_particles, var_names, stride, attr_type, vectorize_velocity)">
  % for var_name in var_names:
          <Attribute Name="${var_name}" AttributeType="${attr_type[var_name]}" Center="Node">
            <DataItem Format="HDF" Dimensions="${n_particles} ${stride[var_name]}" NumberType="float">
              ${file}:/particles/${pname}/arrays/${var_name}
            </DataItem>
          </Attribute>
  % endfor
  % if vectorize_velocity:
          <Attribute Name="V" AttributeType="Vector" Center="Node">
            <DataItem Dimensions="${n_particles} 3" Function="JOIN($0, $1, $2)" ItemType="Function">
              <DataItem Dimensions="${n_particles} 1" Format="HDF" NumberType="Float">
                ${file}:/particles/${pname}/arrays/u
              </DataItem>
              <DataItem Dimensions="${n_particles} 1" Format="HDF" NumberType="Float">
                ${file}:/particles/${pname}/arrays/v
              </DataItem>
              <DataItem Dimensions="${n_particles} 1" Format="HDF" NumberType="Float">
                ${file}:/particles/${pname}/arrays/w
              </DataItem>
            </DataItem>
          </Attribute>
  % endif
</%def>