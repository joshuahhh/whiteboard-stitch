import React from 'react';
import _ from 'underscore';
import ReactBurgerMenu from 'react-burger-menu';
const Menu = ReactBurgerMenu.slide;

import OpenSeadragonViewer from './OpenSeadragonViewer';


const menuStyles = {
  bmBurgerButton: {
    position: 'fixed',
    width: '36px',
    height: '30px',
    left: '36px',
    top: '36px'
  },
  bmBurgerBars: {
    background: '#bdc3c7' //'#373a47'
  },
  bmCrossButton: {
    height: '24px',
    width: '24px'
  },
  bmCross: {
    background: '#666666'
  },
  bmMenu: {
    background: '#dddddd',
    padding: '2.5em 1em 0 0.5em',
    fontSize: '1.15em'
  },
  bmMorphShape: {
    fill: '#373a47'
  },
  bmItemList: {
    color: '#666666',
    padding: '0.8em'
  },
  bmOverlay: {
    background: 'rgba(255, 255, 255, 0.1)'
  }
};

const WhiteboardsViewer = React.createClass({
  getInitialState() {
    const {whiteboards} = this.props;

    const displayedWhiteboard = _.keys(whiteboards)[0];
    const displayedStitching = whiteboards[displayedWhiteboard][0];

    return {displayedWhiteboard, displayedStitching};
  },

  setDisplayed(displayedWhiteboard, displayedStitching) {
    this.setState({displayedWhiteboard, displayedStitching});
  },

  render() {
    const {whiteboards} = this.props;
    const {displayedWhiteboard, displayedStitching} = this.state;

    const dataRoot = 'data/' + displayedWhiteboard + '/' + displayedStitching + '/';
    const tileSources = [
      dataRoot + 'stitched/image.dzi',
      dataRoot + 'homographies/image.dzi',
      dataRoot + 'masks/image.dzi'
    ];

    const whiteboardsInOrder = _.sortBy(_.pairs(whiteboards), 0);

    return (
      <div style={{background: '#444444'}}>
        <Menu styles={menuStyles}>
          <div>
            <div style={{height: '100%', overflow: 'scroll'}}>
              {whiteboardsInOrder.map(([whiteboardName, stitchings]) =>
                <div key={whiteboardName} style={{paddingLeft: 20, textIndent: -20, paddingBottom: 20}}>
                  { stitchings.length == 1
                    ? <a href='#' onClick={() => this.setDisplayed(whiteboardName, stitchings[0])}>
                        {whiteboardName}
                      </a>
                    : <span>
                        {whiteboardName}
                        {stitchings.map((stitchingName, i) =>
                          <a href='#' onClick={() => this.setDisplayed(whiteboardName, stitchingName)}>
                            {' '}[{i + 1}]
                          </a>
                        )}
                      </span>
                  }
                </div>
              )}
              <div>
                <div style={{marginTop: 50}}>
                  <i>
                    Use the left- and right-arrows to switch between the stitched view, the image boundaries view, and the compositing-mask view.
                  </i>
                </div>
              </div>
            </div>
          </div>
        </Menu>
        <OpenSeadragonViewer
          key={displayedWhiteboard + '/' + displayedStitching}
          tileSources={tileSources}
          sequenceMode={true}
          controlPosition='BOTTOM_LEFT'
          style={{width: '100%', height: '100%'}}/>
      </div>
    );
  }
});

export default WhiteboardsViewer;
