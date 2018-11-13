import React from 'react';

import WhiteboardsViewer from './WhiteboardsViewer';

const App = React.createClass({
  getInitialState() {
    return {
      whiteboardsData: null
    };
  },

  componentDidMount() {
    var request = new XMLHttpRequest();
    request.onreadystatechange = () => {
        if (request.readyState == 4 && request.status == 200) {
          this.setState({whiteboardsData: JSON.parse(request.responseText)});
        }
    };
    request.open('GET', 'data/whiteboards.json', true);
    request.send(null);
  },

  render() {
    const {whiteboardsData} = this.state;

    if (whiteboardsData) {
      return <WhiteboardsViewer whiteboards={whiteboardsData} />;
    } else {
      return <div>Loading...</div>;
    }
  }
});

export default App;
