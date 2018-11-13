import React from 'react';
import OpenSeadragon from 'openseadragon';

const OpenSeadragonViewer = React.createClass({
  render() {
    const {tileSources, controlPosition, sequenceMode, ...otherProps} = this.props;

    return <div ref='div' {...otherProps}/>;
  },

  componentDidMount() {
    const {tileSources, controlPosition, sequenceMode} = this.props;

    var navigationControlAnchor = OpenSeadragon.ControlAnchor[controlPosition || 'TOP_LEFT'];

    this.viewer = OpenSeadragon({
      element: this.refs.div,
      prefixUrl: "nav-images/",
      tileSources: tileSources,
      sequenceMode: sequenceMode,
      showNavigator: true,
      navigationControlAnchor: navigationControlAnchor,
      sequenceControlAnchor: navigationControlAnchor,
      preserveViewport: true,
      visibilityRatio: 1.0,
    });
  },
  componentWillUnmount() {
    this.viewer.destroy();
  }
});

export default OpenSeadragonViewer;
